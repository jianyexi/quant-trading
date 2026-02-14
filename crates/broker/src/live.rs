/// Live broker interface — placeholder for real broker API integration (e.g., CTP).

pub struct LiveBroker;

impl LiveBroker {
    pub fn new() -> Self {
        LiveBroker
    }
}

impl Default for LiveBroker {
    fn default() -> Self {
        Self::new()
    }
}
