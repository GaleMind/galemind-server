//! Reading a server's Prometheus metrics.
//!
//! The exposition format is parsed into samples so that a client can watch a
//! number over time, or record a run as a timeseries, without a Prometheus
//! server in between.

use galemind_core::InferError;
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, InferError>;

/// One metric sample: its name, its labels and its value.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

impl Sample {
    /// The value of one label, if present.
    pub fn label(&self, name: &str) -> Option<&str> {
        self.labels.get(name).map(String::as_str)
    }

    /// Whether every given label matches.
    pub fn matches(&self, wanted: &[(&str, &str)]) -> bool {
        wanted
            .iter()
            .all(|(name, value)| self.label(name) == Some(*value))
    }
}

/// Every sample a scrape reported.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scrape {
    pub samples: Vec<Sample>,
}

impl Scrape {
    /// Parses Prometheus text exposition.
    ///
    /// Comments, help and type lines are skipped; a line that cannot be read is
    /// skipped rather than failing the scrape, because one unfamiliar metric
    /// must not stop a client reading the rest.
    pub fn parse(text: &str) -> Self {
        let samples = text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .filter(|line| !line.trim().is_empty())
            .filter_map(parse_line)
            .collect();
        Self { samples }
    }

    /// The first sample of a metric, ignoring labels.
    pub fn value(&self, name: &str) -> Option<f64> {
        self.samples
            .iter()
            .find(|sample| sample.name == name)
            .map(|sample| sample.value)
    }

    /// The sample of a metric carrying the given labels.
    pub fn value_where(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        self.samples
            .iter()
            .find(|sample| sample.name == name && sample.matches(labels))
            .map(|sample| sample.value)
    }

    /// Every sample of one metric.
    pub fn series(&self, name: &str) -> Vec<&Sample> {
        self.samples
            .iter()
            .filter(|sample| sample.name == name)
            .collect()
    }

    /// Sum over every sample of a metric, across labels.
    pub fn total(&self, name: &str) -> f64 {
        self.series(name).iter().map(|sample| sample.value).sum()
    }
}

fn parse_line(line: &str) -> Option<Sample> {
    let (head, value) = line.trim().rsplit_once(' ')?;
    // A value may be a number, or NaN and the infinities Prometheus allows.
    let value = match value {
        "NaN" => f64::NAN,
        "+Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        other => other.parse().ok()?,
    };

    let (name, labels) = match head.split_once('{') {
        Some((name, rest)) => (name, parse_labels(rest.trim_end_matches('}'))),
        None => (head, BTreeMap::new()),
    };

    Some(Sample {
        name: name.trim().to_string(),
        labels,
        value,
    })
}

fn parse_labels(text: &str) -> BTreeMap<String, String> {
    let mut labels = BTreeMap::new();
    let mut rest = text;

    // Label values are quoted and may contain commas, so the text is walked
    // quote by quote rather than split on commas.
    while let Some((name, after)) = rest.split_once('=') {
        let after = after.trim_start();
        let Some(after) = after.strip_prefix('"') else {
            break;
        };
        let Some((value, remainder)) = after.split_once('"') else {
            break;
        };
        labels.insert(
            name.trim().trim_start_matches(',').trim().to_string(),
            value.to_string(),
        );
        rest = remainder;
    }
    labels
}

/// Fetches and parses a server's metrics over HTTP.
pub async fn scrape(endpoint: &str) -> Result<Scrape> {
    let url = format!("{}/metrics", endpoint.trim_end_matches('/'));
    let response = reqwest::get(&url)
        .await
        .map_err(|e| InferError::Internal(format!("{url}: {e}")))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| InferError::Internal(format!("cannot read {url}: {e}")))?;

    if !status.is_success() {
        return Err(InferError::remote_server_error(format!(
            "{url} answered {status}"
        )));
    }
    Ok(Scrape::parse(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPOSITION: &str = "\
# HELP gm_inference_request_success Successful inference requests.
# TYPE gm_inference_request_success counter
gm_inference_request_success{model=\"building_energy\",version=\"1\"} 12
gm_inference_request_success{model=\"building_energy_class\",version=\"1\"} 3
# TYPE gm_cpu_utilization gauge
gm_cpu_utilization 0.250000
gm_memory_usage_bytes 1048576
gm_inference_request_duration_seconds_bucket{model=\"building_energy\",version=\"1\",le=\"+Inf\"} 12
gm_server_info{version=\"0.2.0\"} 1
";

    #[test]
    fn labelled_and_unlabelled_samples_are_both_read() {
        let scrape = Scrape::parse(EXPOSITION);
        assert_eq!(scrape.value("gm_cpu_utilization"), Some(0.25));
        assert_eq!(scrape.value("gm_memory_usage_bytes"), Some(1_048_576.0));
        assert_eq!(
            scrape.value_where(
                "gm_inference_request_success",
                &[("model", "building_energy")]
            ),
            Some(12.0)
        );
    }

    #[test]
    fn help_and_type_lines_are_not_samples() {
        let scrape = Scrape::parse(EXPOSITION);
        assert!(scrape
            .samples
            .iter()
            .all(|sample| !sample.name.starts_with('#')));
        assert_eq!(scrape.samples.len(), 6);
    }

    #[test]
    fn every_series_of_a_metric_can_be_summed() {
        let scrape = Scrape::parse(EXPOSITION);
        assert_eq!(scrape.series("gm_inference_request_success").len(), 2);
        assert_eq!(scrape.total("gm_inference_request_success"), 15.0);
    }

    #[test]
    fn a_histogram_bound_of_infinity_is_read() {
        let scrape = Scrape::parse(EXPOSITION);
        assert_eq!(
            scrape.value_where(
                "gm_inference_request_duration_seconds_bucket",
                &[("le", "+Inf")]
            ),
            Some(12.0)
        );
    }

    #[test]
    fn a_label_value_containing_a_comma_stays_whole() {
        let scrape = Scrape::parse("metric{note=\"a,b\",other=\"c\"} 5\n");
        let sample = &scrape.samples[0];
        assert_eq!(sample.label("note"), Some("a,b"));
        assert_eq!(sample.label("other"), Some("c"));
    }

    #[test]
    fn non_finite_values_are_read_rather_than_dropped() {
        let scrape = Scrape::parse("a NaN\nb +Inf\nc -Inf\n");
        assert!(scrape.value("a").unwrap().is_nan());
        assert_eq!(scrape.value("b"), Some(f64::INFINITY));
        assert_eq!(scrape.value("c"), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn an_unreadable_line_is_skipped_and_the_rest_still_parses() {
        let scrape = Scrape::parse("broken line here\ngood_metric 7\n");
        assert_eq!(scrape.value("good_metric"), Some(7.0));
        assert_eq!(scrape.samples.len(), 1);
    }

    #[test]
    fn an_absent_metric_has_no_value() {
        let scrape = Scrape::parse(EXPOSITION);
        assert_eq!(scrape.value("gm_nonexistent"), None);
        assert_eq!(scrape.total("gm_nonexistent"), 0.0);
    }
}
