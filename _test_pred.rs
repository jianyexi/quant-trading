use quant_strategy::lgb_inference::LightGBMModel;
fn main() {
    let m = LightGBMModel::load("ml_models/factor_model.model").unwrap();
    println!("Trees: {}", m.num_trees());
    // Typical feature values
    let f = [0.01f32, 0.03, 0.05, 0.1, 0.02, 0.03, 0.01, 0.02, 0.03, 0.05, 0.01, 55.0, 0.5, 0.001, 1.2, 0.1, 0.6, 0.002, 0.02, 0.3, 0.2, 0.5, 0.4, 0.005];
    println!("Bullish features: {:.4}", m.predict(&f));
    let f2 = [-0.02f32, -0.05, -0.1, -0.15, 0.04, 0.06, -0.02, -0.03, -0.05, -0.1, -0.03, 25.0, -1.0, -0.002, 0.8, -0.2, 0.2, -0.005, 0.03, 0.4, 0.1, 0.1, 0.3, -0.01];
    println!("Bearish features: {:.4}", m.predict(&f2));
    let f3 = [0.0f32; 24];
    println!("Neutral features: {:.4}", m.predict(&f3));
}
