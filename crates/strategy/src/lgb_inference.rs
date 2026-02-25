// Native LightGBM model inference in Rust.
//
// Parses the LightGBM text model format and evaluates decision trees
// directly — zero external dependencies, zero network latency.
//
// Model format reference: LightGBM text export with `model.save_model()`.

use std::path::Path;

/// A single decision tree node.
#[derive(Debug, Clone)]
enum TreeNode {
    /// Internal split node: feature index, threshold, left child, right child
    Split {
        feature: usize,
        threshold: f64,
        left: usize,  // index into nodes vec
        right: usize,
    },
    /// Leaf node with a value
    Leaf(f64),
}

/// A single decision tree.
#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<TreeNode>,
    root: usize,
    shrinkage: f64,
}

impl Tree {
    /// Evaluate a single tree given feature values.
    fn predict(&self, features: &[f32]) -> f64 {
        let mut idx = self.root;
        loop {
            match &self.nodes[idx] {
                TreeNode::Leaf(val) => return *val,
                TreeNode::Split { feature, threshold, left, right } => {
                    let fval = if *feature < features.len() {
                        features[*feature] as f64
                    } else {
                        0.0
                    };
                    idx = if fval <= *threshold { *left } else { *right };
                }
            }
        }
    }
}

/// A complete LightGBM model loaded from text format.
#[derive(Debug, Clone)]
pub struct LightGBMModel {
    trees: Vec<Tree>,
    #[allow(dead_code)]
    num_class: usize,
    /// Sigmoid objective (binary classification)
    sigmoid: bool,
    feature_names: Vec<String>,
}

impl LightGBMModel {
    /// Load a LightGBM model from a text file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read model file: {}", e))?;
        Self::parse(&content)
    }

    /// Parse a LightGBM model from text content.
    fn parse(content: &str) -> Result<Self, String> {
        let mut num_class = 1usize;
        let mut sigmoid = false;
        let mut feature_names = Vec::new();
        let mut trees = Vec::new();

        // Parse header
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.starts_with("num_class=") {
                num_class = line[10..].parse().unwrap_or(1);
            } else if line.starts_with("objective=") {
                sigmoid = line.contains("sigmoid");
            } else if line.starts_with("feature_names=") {
                feature_names = line[14..].split_whitespace()
                    .map(|s| s.to_string()).collect();
            } else if line.starts_with("Tree=") {
                // Parse a tree block
                i += 1;
                let tree = Self::parse_tree(&lines, &mut i)?;
                trees.push(tree);
                continue; // i already advanced
            }

            i += 1;
        }

        if trees.is_empty() {
            return Err("No trees found in model file".to_string());
        }

        Ok(Self { trees, num_class, sigmoid, feature_names })
    }

    /// Parse a single tree block starting after "Tree=N".
    fn parse_tree(lines: &[&str], i: &mut usize) -> Result<Tree, String> {
        let mut num_leaves = 0usize;
        let mut split_feature: Vec<usize> = Vec::new();
        let mut threshold: Vec<f64> = Vec::new();
        let mut left_child: Vec<i32> = Vec::new();
        let mut right_child: Vec<i32> = Vec::new();
        let mut leaf_value: Vec<f64> = Vec::new();
        let mut shrinkage = 1.0f64;

        while *i < lines.len() {
            let line = lines[*i].trim();

            if line.is_empty() || line.starts_with("Tree=") || line == "end of trees" {
                break;
            }

            if line.starts_with("num_leaves=") {
                num_leaves = line[11..].parse().unwrap_or(0);
            } else if line.starts_with("split_feature=") {
                split_feature = line[14..].split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect();
            } else if line.starts_with("threshold=") {
                threshold = line[10..].split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect();
            } else if line.starts_with("left_child=") {
                left_child = line[11..].split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect();
            } else if line.starts_with("right_child=") {
                right_child = line[12..].split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect();
            } else if line.starts_with("leaf_value=") {
                leaf_value = line[11..].split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect();
            } else if line.starts_with("shrinkage=") {
                shrinkage = line[10..].parse().unwrap_or(1.0);
            }

            *i += 1;
        }

        if num_leaves == 0 {
            return Err("Tree has 0 leaves".to_string());
        }

        // Build node array: internal nodes first, then leaf nodes
        // LightGBM convention: negative child index means leaf (-(idx+1))
        let num_internal = num_leaves - 1;
        let total_nodes = num_internal + num_leaves;
        let mut nodes = Vec::with_capacity(total_nodes);

        // First add internal nodes (will be updated with correct child indices)
        for _ in 0..num_internal {
            nodes.push(TreeNode::Leaf(0.0)); // placeholder
        }
        // Then add leaf nodes
        for lv in &leaf_value {
            nodes.push(TreeNode::Leaf(*lv));
        }

        // Now update internal nodes with correct child pointers
        for j in 0..num_internal {
            if j >= split_feature.len() || j >= threshold.len()
                || j >= left_child.len() || j >= right_child.len() {
                continue;
            }

            let left_idx = if left_child[j] < 0 {
                // Leaf: -(idx+1) → leaf index → offset by num_internal
                num_internal + ((-left_child[j] - 1) as usize)
            } else {
                left_child[j] as usize
            };

            let right_idx = if right_child[j] < 0 {
                num_internal + ((-right_child[j] - 1) as usize)
            } else {
                right_child[j] as usize
            };

            nodes[j] = TreeNode::Split {
                feature: split_feature[j],
                threshold: threshold[j],
                left: left_idx,
                right: right_idx,
            };
        }

        Ok(Tree { nodes, root: 0, shrinkage })
    }

    /// Predict probability for a feature vector.
    /// Returns sigmoid(sum of all tree outputs * shrinkage) for binary classification.
    pub fn predict(&self, features: &[f32]) -> f64 {
        let infer_t0 = std::time::Instant::now();
        let raw_score: f64 = self.trees.iter()
            .map(|t| t.predict(features) * t.shrinkage)
            .sum();

        let result = if self.sigmoid {
            1.0 / (1.0 + (-raw_score).exp()) // sigmoid
        } else {
            raw_score
        };
        let infer_us = infer_t0.elapsed().as_micros();
        tracing::debug!(mode="embedded", latency_us=%infer_us, "ML inference");
        result
    }

    /// Number of trees in the model.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Feature names from the model.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_tree() {
        let model_text = r#"tree
version=v4
num_class=1
num_tree_per_iteration=1
objective=binary sigmoid:1
feature_names=f0 f1

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=100 50
threshold=0.5 0.3
decision_type=2 2
left_child=-1 -2
right_child=1 -3
leaf_value=-0.1 0.2 0.3
leaf_weight=100 200 300
leaf_count=100 200 300
shrinkage=1

end of trees
"#;
        let model = LightGBMModel::parse(model_text).unwrap();
        assert_eq!(model.num_trees(), 1);
        assert!(model.sigmoid);

        // f0 <= 0.5 → leaf -0.1 → sigmoid(-0.1) ≈ 0.475
        let p1 = model.predict(&[0.3, 0.0]);
        assert!((p1 - 0.475).abs() < 0.01, "Got {}", p1);

        // f0 > 0.5, f1 <= 0.3 → leaf 0.2 → sigmoid(0.2) ≈ 0.55
        let p2 = model.predict(&[0.6, 0.1]);
        assert!((p2 - 0.55).abs() < 0.01, "Got {}", p2);

        // f0 > 0.5, f1 > 0.3 → leaf 0.3 → sigmoid(0.3) ≈ 0.574
        let p3 = model.predict(&[0.6, 0.5]);
        assert!((p3 - 0.574).abs() < 0.01, "Got {}", p3);
    }

    #[test]
    fn test_load_real_model() {
        let path = "../../ml_models/factor_model.model";
        if !std::path::Path::new(path).exists() {
            return; // skip if model not present
        }
        let model = LightGBMModel::load(path).unwrap();
        assert!(model.num_trees() > 0, "Should have trees");
        assert!(model.sigmoid, "Should be sigmoid objective");

        // Test prediction with dummy features
        let features = [0.0f32; 24];
        let prob = model.predict(&features);
        assert!(prob >= 0.0 && prob <= 1.0, "Probability out of range: {}", prob);
    }
}
