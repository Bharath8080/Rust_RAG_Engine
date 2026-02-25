use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis};

#[pyclass]
struct RagEngine {
    chunks: Vec<String>,
    embeddings: Option<Array2<f32>>,
    dim: usize,
}

#[pymethods]
impl RagEngine {
    #[new]
    fn new() -> Self {
        RagEngine {
            chunks: Vec::new(),
            embeddings: None,
            dim: 0,
        }
    }

    /// Smart text splitting that respects boundaries (paragraphs, sentences, words)
    fn chunk_text(&mut self, text: String, chunk_size: usize, overlap: usize) -> Vec<String> {
        let separators = vec!["\n\n", "\n", ". ", " ", ""];
        
        fn split_text(text: &str, separators: &[&str], max_size: usize, overlap: usize) -> Vec<String> {
            if text.chars().count() <= max_size || separators.is_empty() {
                return vec![text.to_string()];
            }

            let separator = separators[0];
            let parts: Vec<&str> = text.split(separator).collect();
            let mut final_chunks = Vec::new();
            let mut current_chunk: Vec<&str> = Vec::new();
            let mut current_len = 0;

            for part in parts {
                let part_len = part.chars().count();
                
                if current_len + part_len + separator.chars().count() > max_size {
                    if !current_chunk.is_empty() {
                        let chunk_str = current_chunk.join(separator);
                        final_chunks.push(chunk_str);
                        
                        // Handle overlap: Keep last few words/chars
                        let mut overlap_chunk = Vec::new();
                        let mut overlap_len = 0;
                        for p in current_chunk.iter().rev() {
                            let p_len = p.chars().count();
                            if overlap_len + p_len <= overlap {
                                overlap_chunk.insert(0, *p);
                                overlap_len += p_len;
                            } else {
                                break;
                            }
                        }
                        current_chunk = overlap_chunk;
                        current_len = overlap_len;
                    }
                    
                    if part_len > max_size {
                        final_chunks.extend(split_text(part, &separators[1..], max_size, overlap));
                        // After large part, don't keep overlap to avoid complexity
                        current_chunk = Vec::new();
                        current_len = 0;
                    } else {
                        current_chunk.push(part);
                        current_len += part_len;
                    }
                } else {
                    current_chunk.push(part);
                    current_len += part_len;
                    if !current_chunk.is_empty() { current_len += separator.chars().count(); }
                }
            }
            if !current_chunk.is_empty() {
                final_chunks.push(current_chunk.join(separator));
            }
            final_chunks
        }

        let chunks = split_text(&text, &separators, chunk_size, overlap);
        self.chunks = chunks.clone();
        chunks
    }

    /// Load and pre-normalize embeddings for O(1) similarity calculation
    fn load_embeddings(&mut self, flattened_docs: Vec<f32>, dim: usize) {
        let n_docs = flattened_docs.len() / dim;
        let mut matrix = Array2::from_shape_vec((n_docs, dim), flattened_docs).unwrap();
        
        for mut row in matrix.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-9 { row /= norm; }
        }
        
        self.embeddings = Some(matrix);
        self.dim = dim;
    }

    /// Optimized search using pre-normalized dot products
    fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(String, f32)> {
        let Some(ref docs) = self.embeddings else { return Vec::new(); };
        let mut q = Array1::from(query);
        let q_norm = q.dot(&q).sqrt();
        if q_norm > 1e-9 { q /= q_norm; }

        let scores_arr = docs.dot(&q);
        let mut scores: Vec<(usize, f32)> = scores_arr
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(top_k).map(|(i, s)| (self.chunks[i].clone(), s)).collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RagEngine>()?;
    Ok(())
}