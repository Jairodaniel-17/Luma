use crate::memory::types::{ConstraintOperator, RuleCondition};

pub fn evaluate_condition(condition: &RuleCondition, context: &serde_json::Value) -> bool {
    let Some(actual) = lookup_path(context, &condition.field) else {
        return false;
    };

    match condition.op {
        ConstraintOperator::Eq => actual == &condition.value,
        ConstraintOperator::Neq => actual != &condition.value,
        ConstraintOperator::Gt => compare_f64(actual, &condition.value, |a, b| a > b),
        ConstraintOperator::Gte => compare_f64(actual, &condition.value, |a, b| a >= b),
        ConstraintOperator::Lt => compare_f64(actual, &condition.value, |a, b| a < b),
        ConstraintOperator::Lte => compare_f64(actual, &condition.value, |a, b| a <= b),
        ConstraintOperator::Contains => contains(actual, &condition.value),
        ConstraintOperator::In => in_list(actual, &condition.value),
    }
}

fn lookup_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for part in path.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

fn compare_f64(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
    predicate: impl Fn(f64, f64) -> bool,
) -> bool {
    let Some(left) = actual.as_f64() else {
        return false;
    };
    let Some(right) = expected.as_f64() else {
        return false;
    };
    predicate(left, right)
}

fn contains(actual: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match (actual, expected) {
        (serde_json::Value::Array(items), value) => items.iter().any(|item| item == value),
        (serde_json::Value::String(text), serde_json::Value::String(needle)) => {
            text.contains(needle)
        }
        _ => false,
    }
}

fn in_list(actual: &serde_json::Value, expected: &serde_json::Value) -> bool {
    let serde_json::Value::Array(items) = expected else {
        return false;
    };
    items.iter().any(|item| item == actual)
}

#[cfg(test)]
mod tests {
    use super::evaluate_condition;
    use crate::memory::types::{ConstraintOperator, RuleCondition};

    #[test]
    fn supports_nested_numeric_comparisons() {
        let ctx = serde_json::json!({ "order": { "total": 150.0 } });
        let condition = RuleCondition {
            field: "order.total".to_string(),
            op: ConstraintOperator::Gt,
            value: serde_json::json!(100.0),
        };
        assert!(evaluate_condition(&condition, &ctx));
    }
}
