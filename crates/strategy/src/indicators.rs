// Technical indicators

/// Simple Moving Average
pub struct SMA {
    period: usize,
    values: Vec<f64>,
}

impl SMA {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            values: Vec::with_capacity(period),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.values.push(value);
        if self.values.len() > self.period {
            self.values.remove(0);
        }
        self.value()
    }

    pub fn value(&self) -> Option<f64> {
        if self.values.len() == self.period {
            let sum: f64 = self.values.iter().sum();
            Some(sum / self.period as f64)
        } else {
            None
        }
    }
}

/// Exponential Moving Average
pub struct EMA {
    period: usize,
    multiplier: f64,
    current: Option<f64>,
    count: usize,
    sum: f64,
}

impl EMA {
    pub fn new(period: usize) -> Self {
        let multiplier = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            multiplier,
            current: None,
            count: 0,
            sum: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        match self.current {
            None => {
                self.sum += value;
                if self.count == self.period {
                    let sma = self.sum / self.period as f64;
                    self.current = Some(sma);
                }
            }
            Some(prev) => {
                self.current = Some((value - prev) * self.multiplier + prev);
            }
        }
        self.current
    }

    pub fn value(&self) -> Option<f64> {
        self.current
    }
}

/// MACD (Moving Average Convergence Divergence)
pub struct MACD {
    fast_ema: EMA,
    slow_ema: EMA,
    signal_ema: EMA,
    macd_line: Option<f64>,
    signal_line: Option<f64>,
    histogram: Option<f64>,
}

impl MACD {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_ema: EMA::new(fast),
            slow_ema: EMA::new(slow),
            signal_ema: EMA::new(signal),
            macd_line: None,
            signal_line: None,
            histogram: None,
        }
    }

    pub fn update(&mut self, value: f64) {
        let fast_val = self.fast_ema.update(value);
        let slow_val = self.slow_ema.update(value);

        if let (Some(f), Some(s)) = (fast_val, slow_val) {
            let macd = f - s;
            self.macd_line = Some(macd);

            if let Some(sig) = self.signal_ema.update(macd) {
                self.signal_line = Some(sig);
                self.histogram = Some(macd - sig);
            }
        }
    }

    pub fn macd_line(&self) -> Option<f64> {
        self.macd_line
    }

    pub fn signal_line(&self) -> Option<f64> {
        self.signal_line
    }

    pub fn histogram(&self) -> Option<f64> {
        self.histogram
    }
}

/// RSI (Relative Strength Index)
pub struct RSI {
    period: usize,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    prev_value: Option<f64>,
    count: usize,
    gains: Vec<f64>,
    losses: Vec<f64>,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: None,
            avg_loss: None,
            prev_value: None,
            count: 0,
            gains: Vec::with_capacity(period),
            losses: Vec::with_capacity(period),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        if let Some(prev) = self.prev_value {
            let change = value - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            self.count += 1;

            match self.avg_gain {
                None => {
                    self.gains.push(gain);
                    self.losses.push(loss);

                    if self.count == self.period {
                        let avg_g: f64 = self.gains.iter().sum::<f64>() / self.period as f64;
                        let avg_l: f64 = self.losses.iter().sum::<f64>() / self.period as f64;
                        self.avg_gain = Some(avg_g);
                        self.avg_loss = Some(avg_l);
                    }
                }
                Some(prev_avg_gain) => {
                    let p = self.period as f64;
                    let avg_g = (prev_avg_gain * (p - 1.0) + gain) / p;
                    let avg_l = (self.avg_loss.unwrap() * (p - 1.0) + loss) / p;
                    self.avg_gain = Some(avg_g);
                    self.avg_loss = Some(avg_l);
                }
            }
        }
        self.prev_value = Some(value);
        self.value()
    }

    pub fn value(&self) -> Option<f64> {
        match (self.avg_gain, self.avg_loss) {
            (Some(avg_gain), Some(avg_loss)) => {
                if avg_loss == 0.0 {
                    Some(100.0)
                } else {
                    let rs = avg_gain / avg_loss;
                    Some(100.0 - 100.0 / (1.0 + rs))
                }
            }
            _ => None,
        }
    }
}

/// Bollinger Bands
pub struct BollingerBands {
    sma: SMA,
    period: usize,
    std_dev_multiplier: f64,
    values: Vec<f64>,
    upper: Option<f64>,
    middle: Option<f64>,
    lower: Option<f64>,
}

impl BollingerBands {
    pub fn new(period: usize, std_dev: f64) -> Self {
        Self {
            sma: SMA::new(period),
            period,
            std_dev_multiplier: std_dev,
            values: Vec::with_capacity(period),
            upper: None,
            middle: None,
            lower: None,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.values.push(value);
        if self.values.len() > self.period {
            self.values.remove(0);
        }

        if let Some(mid) = self.sma.update(value) {
            self.middle = Some(mid);

            let variance = self.values.iter().map(|v| (v - mid).powi(2)).sum::<f64>()
                / self.period as f64;
            let std_dev = variance.sqrt();

            self.upper = Some(mid + self.std_dev_multiplier * std_dev);
            self.lower = Some(mid - self.std_dev_multiplier * std_dev);
        }
    }

    pub fn upper(&self) -> Option<f64> {
        self.upper
    }

    pub fn middle(&self) -> Option<f64> {
        self.middle
    }

    pub fn lower(&self) -> Option<f64> {
        self.lower
    }
}

/// KDJ indicator (popular in Chinese market)
pub struct KDJ {
    period: usize,
    k_period: usize,
    d_period: usize,
    highs: Vec<f64>,
    lows: Vec<f64>,
    k: Option<f64>,
    d: Option<f64>,
    j: Option<f64>,
}

impl KDJ {
    pub fn new(period: usize, k_period: usize, d_period: usize) -> Self {
        Self {
            period,
            k_period,
            d_period,
            highs: Vec::with_capacity(period),
            lows: Vec::with_capacity(period),
            k: None,
            d: None,
            j: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) {
        self.highs.push(high);
        self.lows.push(low);
        if self.highs.len() > self.period {
            self.highs.remove(0);
            self.lows.remove(0);
        }

        if self.highs.len() < self.period {
            return;
        }

        let highest = self.highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = self.lows.iter().cloned().fold(f64::INFINITY, f64::min);

        let rsv = if (highest - lowest).abs() < f64::EPSILON {
            50.0
        } else {
            (close - lowest) / (highest - lowest) * 100.0
        };

        // K = 2/3 * prev_K + 1/3 * RSV (smoothing with k_period)
        let smooth_k = 1.0 / self.k_period as f64;
        let prev_k = self.k.unwrap_or(50.0);
        let k = prev_k * (1.0 - smooth_k) + rsv * smooth_k;

        let smooth_d = 1.0 / self.d_period as f64;
        let prev_d = self.d.unwrap_or(50.0);
        let d = prev_d * (1.0 - smooth_d) + k * smooth_d;

        let j = 3.0 * k - 2.0 * d;

        self.k = Some(k);
        self.d = Some(d);
        self.j = Some(j);
    }

    pub fn k(&self) -> Option<f64> {
        self.k
    }

    pub fn d(&self) -> Option<f64> {
        self.d
    }

    pub fn j(&self) -> Option<f64> {
        self.j
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let mut sma = SMA::new(3);
        assert_eq!(sma.update(1.0), None);
        assert_eq!(sma.update(2.0), None);
        assert_eq!(sma.update(3.0), Some(2.0)); // (1+2+3)/3
        assert_eq!(sma.update(4.0), Some(3.0)); // (2+3+4)/3
        assert_eq!(sma.update(5.0), Some(4.0)); // (3+4+5)/3
    }

    #[test]
    fn test_ema_basic() {
        let mut ema = EMA::new(3);
        // First 3 values seed the SMA: (2+4+6)/3 = 4.0
        assert_eq!(ema.update(2.0), None);
        assert_eq!(ema.update(4.0), None);
        let v = ema.update(6.0).unwrap();
        assert!((v - 4.0).abs() < 1e-10);

        // multiplier = 2/(3+1) = 0.5
        // EMA = (8 - 4) * 0.5 + 4 = 6.0
        let v = ema.update(8.0).unwrap();
        assert!((v - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_basic() {
        let mut rsi = RSI::new(5);
        // Need 6 values (1 baseline + 5 changes) to get first RSI
        let prices = [44.0, 44.25, 44.5, 43.75, 44.5, 44.25];
        for &p in &prices[..5] {
            assert_eq!(rsi.update(p), None);
        }
        let val = rsi.update(prices[5]);
        assert!(val.is_some());
        let rsi_val = val.unwrap();
        assert!(rsi_val > 0.0 && rsi_val < 100.0);
    }

    #[test]
    fn test_rsi_all_gains() {
        let mut rsi = RSI::new(3);
        // Steadily increasing: all gains, no losses => RSI = 100
        rsi.update(1.0);
        rsi.update(2.0);
        rsi.update(3.0);
        let val = rsi.update(4.0).unwrap();
        assert!((val - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_bollinger_bands() {
        let mut bb = BollingerBands::new(3, 2.0);
        bb.update(10.0);
        bb.update(11.0);
        assert!(bb.middle().is_none());
        bb.update(12.0);
        let mid = bb.middle().unwrap();
        assert!((mid - 11.0).abs() < 1e-10);
        assert!(bb.upper().unwrap() > mid);
        assert!(bb.lower().unwrap() < mid);
    }

    #[test]
    fn test_kdj_basic() {
        let mut kdj = KDJ::new(9, 3, 3);
        // Feed 9 bars to initialize
        let data: Vec<(f64, f64, f64)> = vec![
            (10.0, 8.0, 9.0),
            (11.0, 9.0, 10.0),
            (12.0, 10.0, 11.0),
            (13.0, 11.0, 12.0),
            (14.0, 12.0, 13.0),
            (15.0, 13.0, 14.0),
            (16.0, 14.0, 15.0),
            (17.0, 15.0, 16.0),
            (18.0, 16.0, 17.0),
        ];
        for &(h, l, c) in &data[..8] {
            kdj.update(h, l, c);
            assert!(kdj.k().is_none());
        }
        let (h, l, c) = data[8];
        kdj.update(h, l, c);
        assert!(kdj.k().is_some());
        assert!(kdj.d().is_some());
        assert!(kdj.j().is_some());
    }

    #[test]
    fn test_macd_basic() {
        let mut macd = MACD::new(3, 5, 3);
        // Need at least 5 values for slow EMA to initialize
        for i in 1..=5 {
            macd.update(i as f64);
        }
        // After 5 values, MACD line should be available
        assert!(macd.macd_line().is_some());
    }
}
