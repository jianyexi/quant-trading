/// Order state machine for managing order lifecycle transitions.

use quant_core::types::OrderStatus;

pub struct OrderStateMachine;

impl OrderStateMachine {
    /// Check if a transition from one status to another is valid.
    pub fn can_transition(from: &OrderStatus, to: &OrderStatus) -> bool {
        matches!(
            (from, to),
            (OrderStatus::Pending, OrderStatus::Submitted)
                | (OrderStatus::Pending, OrderStatus::Rejected)
                | (OrderStatus::Pending, OrderStatus::Cancelled)
                | (OrderStatus::Submitted, OrderStatus::PartiallyFilled)
                | (OrderStatus::Submitted, OrderStatus::Filled)
                | (OrderStatus::Submitted, OrderStatus::Cancelled)
                | (OrderStatus::Submitted, OrderStatus::Rejected)
                | (OrderStatus::PartiallyFilled, OrderStatus::Filled)
                | (OrderStatus::PartiallyFilled, OrderStatus::Cancelled)
        )
    }

    /// Attempt a state transition, returning the new status or an error.
    pub fn transition(current: &OrderStatus, to: OrderStatus) -> Result<OrderStatus, String> {
        if Self::can_transition(current, &to) {
            Ok(to)
        } else {
            Err(format!(
                "Invalid order status transition: {:?} -> {:?}",
                current, to
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        assert!(OrderStateMachine::can_transition(&OrderStatus::Pending, &OrderStatus::Submitted));
        assert!(OrderStateMachine::can_transition(&OrderStatus::Submitted, &OrderStatus::Filled));
        assert!(OrderStateMachine::can_transition(
            &OrderStatus::PartiallyFilled,
            &OrderStatus::Filled
        ));
    }

    #[test]
    fn test_invalid_transitions() {
        assert!(!OrderStateMachine::can_transition(&OrderStatus::Filled, &OrderStatus::Pending));
        assert!(!OrderStateMachine::can_transition(
            &OrderStatus::Cancelled,
            &OrderStatus::Submitted
        ));
    }

    #[test]
    fn test_transition_ok() {
        let result = OrderStateMachine::transition(&OrderStatus::Pending, OrderStatus::Submitted);
        assert_eq!(result.unwrap(), OrderStatus::Submitted);
    }

    #[test]
    fn test_transition_err() {
        let result = OrderStateMachine::transition(&OrderStatus::Filled, OrderStatus::Pending);
        assert!(result.is_err());
    }
}
