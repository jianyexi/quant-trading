// ONNX Runtime inference for all model types.
//
// Uses the `ort` crate to load any ONNX model (LightGBM, XGBoost,
// CatBoost, LSTM, Transformer) and run inference directly in Rust.
//
// This module is only compiled when the `onnx` feature is enabled.
// Enable with: cargo build --features onnx

use std::path::Path;
use std::sync::Mutex;
use ort::session::Session;
use ort::value::Tensor;
use ndarray::Array2;

/// Universal ONNX model for ML inference.
pub struct OnnxModel {
    session: Mutex<Session>,
}

impl OnnxModel {
    /// Load an ONNX model from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Session builder: {}", e))?
            .commit_from_file(path.as_ref())
            .map_err(|e| format!("Load ONNX: {}", e))?;

        tracing::info!(
            "ONNX model loaded: inputs={}, outputs={}",
            session.inputs().len(),
            session.outputs().len()
        );

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run prediction on a feature vector. Returns probability [0, 1].
    pub fn predict(&self, features: &[f32]) -> Result<f64, String> {
        let n_features = features.len();
        let input = Array2::from_shape_vec((1, n_features), features.to_vec())
            .map_err(|e| format!("Array shape: {}", e))?;
        let tensor = Tensor::from_array(input)
            .map_err(|e| format!("Tensor: {}", e))?;

        let mut session = self.session.lock().unwrap();

        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| format!("Run: {}", e))?;

        // LightGBM/XGBoost ONNX: output[1] = class probabilities [N, 2]
        if outputs.len() >= 2 {
            if let Ok(view) = outputs[1].try_extract_array::<f32>() {
                if view.len() >= 2 {
                    return Ok(view[[0, 1]] as f64);
                }
                return Ok(view[[0, 0]] as f64);
            }
            if let Ok(view) = outputs[1].try_extract_array::<f64>() {
                if view.len() >= 2 {
                    return Ok(view[[0, 1]]);
                }
                return Ok(view[[0, 0]]);
            }
        }

        // Single output: raw score or logit
        if let Ok(view) = outputs[0].try_extract_array::<f32>() {
            let val = view[[0, 0]] as f64;
            return Ok(1.0 / (1.0 + (-val).exp()));
        }

        Err("Cannot extract prediction from ONNX output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_onnx_if_exists() {
        let paths = ["../../ml_models/factor_model.onnx"];
        for path in &paths {
            if Path::new(path).exists() {
                let model = OnnxModel::load(path);
                assert!(model.is_ok(), "Failed to load {}: {:?}", path, model.err());

                let features = vec![0.0f32; 24];
                let prob = model.unwrap().predict(&features);
                assert!(prob.is_ok(), "Prediction failed: {:?}", prob.err());
                let p = prob.unwrap();
                assert!(p >= 0.0 && p <= 1.0, "Probability out of range: {}", p);
                return;
            }
        }
    }
}
