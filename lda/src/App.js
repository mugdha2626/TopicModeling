import React, { useState } from "react";
import PropTypes from "prop-types";
import { useDropzone } from "react-dropzone";
import JSZip from "jszip";
import { saveAs } from "file-saver";
import Footer from "./Footer";

const FileUpload = ({ getRootProps, getInputProps, file }) => (
  <div {...getRootProps()} style={styles.dropzone}>
    <input {...getInputProps()} />
    <p>
      {file
        ? file.name
        : "Drag & drop a ZIP file here, or click to select one"}
    </p>
  </div>
);

const HelpTooltip = ({ text }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <span style={styles.tooltipContainer}>
      <button 
        style={styles.tooltipIcon}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={(e) => {
          e.preventDefault();
          setShowTooltip(!showTooltip);
        }}
      >
        ?
      </button>
      {showTooltip && (
        <div style={styles.tooltipContent}>
          {text}
        </div>
      )}
    </span>
  );
};

const WelcomeOverlay = ({ onDismiss }) => (
  <div style={styles.overlay}>
    <div style={styles.overlayContent}>
      <h2>Welcome to PDF Topic Explorer! 🚀</h2>
      <p>Here's how to get started:</p>
      <ol style={styles.overlayList}>
        <li>📁 Upload a ZIP file containing your PDF documents</li>
        <li>⚙️ Adjust analysis settings using the guidance tooltips</li>
        <li>🔍 Click "Run Analysis" to discover document themes</li>
        <li>📈 Explore results and export your findings</li>
      </ol>
      <button 
        style={styles.overlayButton}
        onClick={onDismiss}
      >
        Get Started!
      </button>
    </div>
  </div>
);

FileUpload.propTypes = {
  getRootProps: PropTypes.func.isRequired,
  getInputProps: PropTypes.func.isRequired,
  file: PropTypes.object,
};

const ModelSelection = ({ modelType, setModelType }) => (
  <div style={styles.modelSelectionContainer}>
    <h3 style={styles.sectionHeader}>
      Analysis Model Selection
      <HelpTooltip text="Choose between LDA (fixed number of topics) or HDP (automatic topic discovery)" />
    </h3>
    <div style={styles.modelOptions}>
      <label style={styles.modelOption}>
        <input
          type="radio"
          name="modelType"
          value="LDA"
          checked={modelType === "LDA"}
          onChange={(e) => setModelType(e.target.value)}
          style={styles.radioInput}
        />
        <span style={styles.modelLabel}>LDA (Latent Dirichlet Allocation)</span>
        <HelpTooltip text="Classic topic modeling: You specify the number of topics. Best for known domain structure." />
      </label>
      <label style={styles.modelOption}>
        <input
          type="radio"
          name="modelType"
          value="HDP"
          checked={modelType === "HDP"}
          onChange={(e) => setModelType(e.target.value)}
          style={styles.radioInput}
        />
        <span style={styles.modelLabel}>HDP (Hierarchical Dirichlet Process)</span>
        <HelpTooltip text="Automatic topic discovery: No need to specify topic count. Better for exploratory analysis." />
      </label>
    </div>
  </div>
);

ModelSelection.propTypes = {
  modelType: PropTypes.string.isRequired,
  setModelType: PropTypes.func.isRequired,
};

const LDASettings = ({
  numTopics,
  setNumTopics,
  numWords,
  setNumWords,
}) => (
  <div style={styles.settingsContainer}>
    {[
      {
        label: "Number of Topics:  ",
        value: numTopics,
        setValue: setNumTopics,
        help: "Optimal range: 15-20 topics. Start low and increase if themes seem too broad. Max of 45."
      },
      {
        label: "Number of Words per Topic:  ",
        value: numWords,
        setValue: setNumWords,
        help: "5-10 words typically provide the best balance of specificity and coverage. Max of 45."
      },
    ].map(({ label, value, setValue, help }, index) => (
      <div key={index} style={styles.setting}>
        <label>
          {label}
          <HelpTooltip text={help} />
          <input
            type="number"
            value={value}
            onChange={(e) => {
              setValue(Math.min(45, Math.max(0, parseInt(e.target.value, 10) || 0)));
            }}
            min="0"
            max="45"
            style={styles.input}
          />
        </label>
      </div>
    ))}
  </div>
);

LDASettings.propTypes = {
  numTopics: PropTypes.number.isRequired,
  setNumTopics: PropTypes.func.isRequired,
  numWords: PropTypes.number.isRequired,
  setNumWords: PropTypes.func.isRequired,
};

const HDPSettings = ({
  numWords,
  setNumWords,
}) => (
  <div style={styles.settingsContainer}>
    <div style={styles.hdpInfoBox}>
      <h4 style={styles.hdpInfoTitle}>ℹ️ HDP Automatic Topic Discovery</h4>
      <p style={styles.hdpInfoText}>
        HDP automatically determines the optimal number of topics based on your corpus.
        You don't need to specify how many topics to find - the model will discover them for you!
      </p>
    </div>
    <div style={styles.setting}>
      <label>
        Words to Display per Topic:
        <HelpTooltip text="How many top words to show in results. This is just for visualization - HDP finds all important words automatically." />
        <input
          type="number"
          value={numWords}
          onChange={(e) => {
            setNumWords(Math.min(45, Math.max(1, parseInt(e.target.value, 10) || 10)));
          }}
          min="1"
          max="45"
          style={styles.input}
        />
      </label>
    </div>
  </div>
);

HDPSettings.propTypes = {
  numWords: PropTypes.number.isRequired,
  setNumWords: PropTypes.func.isRequired,
};

const BibliographySettings = ({
  includeBibliography,
  setIncludeBibliography,
  includeDecadeAnalysis,
  setIncludeDecadeAnalysis,
  stopwords,
  setStopwords,
  numTopPapers,
  setNumTopPapers
}) => (
  <div style={styles.bibliographyContainer}>
    <br />
    <h3 style={styles.sectionHeader}>
      Analysis Preferences
      <HelpTooltip text="Fine-tune how we process your documents" />
    </h3>
    <label style={styles.checkboxLabel}>
      <input
        type="checkbox"
        checked={includeBibliography}
        onChange={(e) => setIncludeBibliography(e.target.checked)}
        style={styles.checkbox}
      />
      Include bibliography sections 📚
      <HelpTooltip text="When checked, keeps bibliography/references sections in analysis. When unchecked, removes them (recommended for cleaner topics)." />
    </label>
    <br />
    <label style={styles.checkboxLabel}>
      <input
        type="checkbox"
        checked={includeDecadeAnalysis}
        onChange={(e) => setIncludeDecadeAnalysis(e.target.checked)}
        style={styles.checkbox}
      />
      Enable temporal analysis 🕰️
      <HelpTooltip text="Requires year in filenames (e.g., 'paper-2015.pdf'). Shows trends over time." />
    </label>
    <br />
    <div style={styles.setting}>
      <label>
        <br />
        Additional Stopwords (comma-separated) 🚫:
        <HelpTooltip text="Add domain-specific terms you want to exclude (e.g., 'participant', 'methodology')" />
        <input
          type="text"
          value={stopwords}
          onChange={(e) => setStopwords(e.target.value)}
          placeholder="e.g., example, word, stop"
          style={styles.stopwordsInput}
        />
      </label>
    </div>
    <br></br>
    <div style={styles.setting}>
      <label>
        Number of Top Papers per Topic:
        <HelpTooltip text="Shows the top papers that contribute to shaping a topic understanding." />
        <input
          type="number"
          value={numTopPapers}
          onChange={(e) => setNumTopPapers(Math.max(1, parseInt(e.target.value, 10) || 5))}
          min="1"
          max="20"
          style={styles.input}
          
        />
      </label>
    </div>
    <br></br>
  </div>
);

BibliographySettings.propTypes = {
  includeBibliography: PropTypes.bool.isRequired,
  setIncludeBibliography: PropTypes.func.isRequired,
  includeDecadeAnalysis: PropTypes.bool.isRequired,
  setIncludeDecadeAnalysis: PropTypes.func.isRequired,
  stopwords: PropTypes.string.isRequired,
  setStopwords: PropTypes.func.isRequired,
  numTopPapers: PropTypes.number.isRequired,
  setNumTopPapers: PropTypes.func.isRequired,
};

const App = () => {
  // Model selection state
  const [modelType, setModelType] = useState("LDA"); // "LDA" or "HDP"
  
  // LDA parameters
  const [numTopics, setNumTopics] = useState(5);
  const [numWords, setNumWords] = useState(10); 
  const [numTopPapers, setNumTopPapers] = useState(5);
  
  // HDP uses automatic parameter tuning
  
  // Common state
  const [file, setFile] = useState(null); 
  const [results, setResults] = useState(null); 
  const [chartBase64, setChartBase64] = useState({});
  const [decadeChartBase64, setDecadeChartBase64] = useState({});
  const [stopwords, setStopwords] = useState(""); 
  const [includeBibliography, setIncludeBibliography] = useState(false); 
  const [includeDecadeAnalysis, setIncludeDecadeAnalysis] = useState(false); 
  const [loading, setLoading] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState("");
  const [topics, setTopics] = useState([]);
  const [showWelcome, setShowWelcome] = useState(true);
  const [showExplanation, setShowExplanation] = useState(false);
  const [comparisonFile1, setComparisonFile1] = useState(null);
  const [comparisonFile2, setComparisonFile2] = useState(null);
  const [prevalenceFile1, setPrevalenceFile1] = useState(null);
  const [prevalenceFile2, setPrevalenceFile2] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const handleTopicSelect = (event) => {
    const selectedTopicId = event.target.value;
    const topic = topics.find((topic) => topic.Topic === selectedTopicId);
    setSelectedTopic(topic); 
  };
  const averageLift = results?.average_lift_per_topic
  ? (results.average_lift_per_topic.reduce((a, b) => a + b, 0) / results.average_lift_per_topic.length).toFixed(1)
  : 'N/A';
  const handleExportCharts = async () => {
    if (!chartBase64 && !decadeChartBase64) {
      alert("No charts available to export!");
      return;
    }

    const zip = new JSZip();
    const folder = zip.folder("topic_analysis_charts");

    if (chartBase64) {
      Object.entries(chartBase64).forEach(([topicName, base64]) => {
        const content = base64.split(";base64,").pop();
        folder.file(`${topicName.replace(/ /g, "_")}.png`, content, {
          base64: true,
        });
      });
    }

    if (decadeChartBase64) {
      const decadeContent = decadeChartBase64.split(";base64,").pop();
      folder.file("decade_analysis.png", decadeContent, { base64: true });
    }

    zip.generateAsync({ type: "blob" }).then((content) => {
      saveAs(content, "topic_analysis_charts.zip");
    });
  };

  const explanationContent = (
    <div style={styles.explanationBox}>
      <button
        style={styles.closeButton}
        onClick={() => setShowExplanation(false)}
      >
        ×
      </button>
      <div style={styles.explanationContent}>
        <p style={styles.explanationText}>
          <strong>Enhanced Word Importance Calculation:</strong>
          <br />
          Combined Score = (0.5 × Raw Score) + (0.3 × TF-IDF) + (0.2 × Saliency)
        </p>
        <ul style={styles.explanationList}>
          <li>
            <strong>Raw Score:</strong> Basic importance from LDA model
          </li>
          <li>
            <strong>TF-IDF:</strong> Term frequency adjusted for cross-topic
            rarity
          </li>
          <li>
            <strong>Saliency:</strong> Balances frequency and distinctiveness
          </li>
          <li>
            <strong>Lift:</strong> Specificity ratio (topic vs global
            probability)
          </li>
          <li>
            <strong>Entropy:</strong> Measures topic concentration (lower = more
            specific)
          </li>
        </ul>
      </div>
      <br></br>
      <div style={styles.explanationContent}>
        <p style={styles.explanationText}>
          This analysis processed {results?.num_pdfs || 0} PDF documents,
          identifying {results?.num_topics || 0} key topics with{" "}
          {results?.num_words || 0} words per topic.
        </p>
        <p style={styles.explanationText}>Key Metrics:</p>
        <ul style={styles.explanationList}>
          <li>📄 Processed Documents: {results?.num_pdfs  || 0}</li>
          <li>🗂️ Identified Topics: {results?.num_topics || 0}</li>
          <li>🔠 Words per Topic: {results?.num_words || 0}</li>
          <li>⏳ Time Period: {results?.time_period || 0}</li>
          <li>
            ⚖️ Average Lift:{" "}
            {averageLift}
          </li>
        </ul>
      </div>
      <br></br>
      <br></br>
      <div style={styles.explanationContent}>
      <p style={styles.explanationText}>
        <strong>Current Model Configuration:</strong>
      </p>
      <ul style={styles.explanationList}>
        <li>
          🧮 Vectorization Parameters:
          <ul>
            <li>max_df: {results?.vectorizer_params?.max_df || 0.95}</li>
            <li>min_df: {results?.vectorizer_params?.min_df || 1}</li>
            <li>Stopwords: {results?.vectorizer_params?.stopwords_count || 'Custom Set'}</li>
          </ul>
        </li>
        <li>
          📉 Model Loss (NLL): {results?.model_loss?.toFixed(2) || 'N/A'}
          <br/><em>(Negative Log Likelihood, lower = better)</em>
        </li>
      </ul>
    </div>
    </div>
  );

  const { getRootProps, getInputProps } = useDropzone({
    accept: ".zip", // Allow only ZIP file uploads
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]); // Set the file when dropped
    },
  });

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please upload a ZIP file.");
      return;
    }

    // Set loading state before making the request
    setLoading(true);

    // Clear previous results to reset state for the new analysis
    setResults(null);
    setChartBase64(null);
    setSelectedTopic(""); // Reset selected topic

    // Clean and log the stopwords
    const cleanedStopwords = stopwords.trim();
    const stopwordsToSend = cleanedStopwords || ""; // Ensure it's always a string

    console.log("Stopwords input:", stopwords);
    console.log("Stopwords to send:", stopwordsToSend);
    console.log("Model type:", modelType);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("numWords", numWords);
    formData.append("numTopPapers", numTopPapers);
    formData.append("stopwords", stopwordsToSend); // Send stopwords to the backend
    formData.append("include_bibliography", includeBibliography);
    formData.append("include_decade_analysis", includeDecadeAnalysis); // Send decade analysis state

    // Add model-specific parameters
    if (modelType === "LDA") {
      formData.append("numTopics", numTopics);
    }
    // HDP uses automatic parameter tuning - no user input needed

    // Choose endpoint based on model type
    const endpoint = modelType === "HDP" ? "http://localhost:5001/analyze-hdp" : "http://localhost:5001/analyze";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Update the state with the results from the backend
      if (data.topics) {
        setTopics(data.topics);
        setResults(data);
      }

      console.log(
        "Received stopwords from backend:",
        data.additional_stopwords
      );

      if (data.topic_charts) {
        setChartBase64(data.topic_charts); // Store ALL topic charts
      }
      if (data.decade_chart_base64) {
        setDecadeChartBase64(data.decade_chart_base64);
      }
    } catch (error) {
      console.error("Error during analysis:", error);
      alert(`Analysis failed: ${error.message}`);
      setResults(null);
      setChartBase64(null);
    } finally {
      setLoading(false);
    }
  };

  
const handleExportCSV = () => {
  if (!results?.topics || !results?.top_papers) {
    alert("No results to export!");
    return;
  }

  const csvRows = [];
  // CSV headers
  csvRows.push([
    "Topic ID",
    "Topic Name",
    "Top Words",
    "Paper Title",
    "Authors",
    "Year",
    "Loading Factor",
    "PubMed ID",
    "Raw Score"
  ].join(','));

  // Add data rows
  results.topics.forEach((topic, topicIdx) => {
    const papers = results.top_papers[topicIdx];
    papers.forEach((paper) => {
      const row = [
        topicIdx + 1,
        `"${topic.Topic}"`,
        `"${topic.Words}"`,
        `"${paper.title.replace(/"/g, '""')}"`,
        `"${paper.author.replace(/"/g, '""')}"`,
        paper.year,
        paper.loading_factor.toFixed(4),
        paper.pubmed_id || 'N/A',
        paper.raw_score.toFixed(4)
      ].join(',');
      csvRows.push(row);
    });
  });

  // Create download
  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "topic_analysis_full.csv";
  a.click();
  window.URL.revokeObjectURL(url);
};

const handleExportTopicDistributions = () => {
  if (!results?.topic_word_distributions) {
    alert("No topic distributions available to export!");
    return;
  }

  const csvRows = [];
  // CSV headers
  csvRows.push(['Topic_ID', 'Topic_Name', 'Word', 'Probability'].join(','));

  // Add data rows
  results.topic_word_distributions.forEach((topic) => {
    Object.entries(topic.word_probabilities).forEach(([word, prob]) => {
      const row = [
        topic.topic_id + 1,
        `"${topic.topic_name}"`,
        `"${word}"`,
        prob.toFixed(8)
      ].join(',');
      csvRows.push(row);
    });
  });

  // Create download
  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "topic_word_distributions.csv";
  a.click();
  window.URL.revokeObjectURL(url);
};

const handleExportTopicPrevalence = () => {
  if (!results?.topic_prevalence) {
    alert("No topic prevalence data available to export!");
    return;
  }

  const csvRows = [];
  // CSV headers
  csvRows.push(['Topic_ID', 'Topic_Name', 'Prevalence'].join(','));

  // Add data rows
  results.topic_prevalence.forEach((topic) => {
    const row = [
      topic.topic_id + 1,
      `"${topic.topic_name}"`,
      topic.prevalence.toFixed(8)
    ].join(',');
    csvRows.push(row);
  });

  // Create download
  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "topic_prevalence.csv";
  a.click();
  window.URL.revokeObjectURL(url);
};

const handleCompareTopics = async () => {
  if (!comparisonFile1 || !comparisonFile2) {
    alert("Please select both topic distribution CSV files for comparison.");
    return;
  }

  setIsComparing(true);
  setComparisonResults(null);

  const formData = new FormData();
  formData.append("file1", comparisonFile1);
  formData.append("file2", comparisonFile2);

  // Add prevalence files if provided (optional)
  if (prevalenceFile1) {
    formData.append("prevalence_file1", prevalenceFile1);
  }
  if (prevalenceFile2) {
    formData.append("prevalence_file2", prevalenceFile2);
  }

  // Optional: add number of bootstrap samples (default 1000)
  formData.append("n_bootstrap", 1000);

  try {
    const response = await fetch("http://localhost:5001/compare", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (data.error) {
      alert(`Comparison failed: ${data.error}`);
    } else {
      setComparisonResults(data);
      console.log("Comparison results:", data);
    }
  } catch (error) {
    console.error("Error during comparison:", error);
    alert(`Comparison failed: ${error.message}`);
  } finally {
    setIsComparing(false);
  }
};

const OptimalTransportDisplay = ({ otResults, hasPrevalence }) => {
  if (!otResults) return null;

  const { distance, bootstrap } = otResults;
  const ciWidth = bootstrap ? bootstrap.ci_upper - bootstrap.ci_lower : 0;

  return (
    <div style={styles.otContainer}>
      <h3 style={styles.metricTitle}>
        🌍 Optimal Transport (OT) Distance
        <HelpTooltip text="Global divergence metric measuring the minimum cost to transform one topic distribution into another. Lower values indicate more similar topic structures." />
      </h3>

      <div style={styles.otMainMetric}>
        <div style={styles.otDistanceBox}>
          <div style={styles.otDistanceValue}>{distance?.toFixed(4) || 'N/A'}</div>
          <div style={styles.otDistanceLabel}>OT Distance</div>
          {!hasPrevalence && (
            <div style={styles.otWarning}>⚠️ Using uniform weights (no prevalence data)</div>
          )}
        </div>

        {bootstrap && (
          <div style={styles.otBootstrapInfo}>
            <h4 style={styles.bootstrapTitle}>95% Confidence Interval (1000 samples)</h4>
            <div style={styles.bootstrapStats}>
              <div style={styles.bootstrapStat}>
                <span style={styles.statLabel}>Mean:</span>
                <span style={styles.statValue}>{bootstrap.mean.toFixed(4)}</span>
              </div>
              <div style={styles.bootstrapStat}>
                <span style={styles.statLabel}>CI Lower:</span>
                <span style={styles.statValue}>{bootstrap.ci_lower.toFixed(4)}</span>
              </div>
              <div style={styles.bootstrapStat}>
                <span style={styles.statLabel}>CI Upper:</span>
                <span style={styles.statValue}>{bootstrap.ci_upper.toFixed(4)}</span>
              </div>
              <div style={styles.bootstrapStat}>
                <span style={styles.statLabel}>Std Dev:</span>
                <span style={styles.statValue}>{bootstrap.std.toFixed(4)}</span>
              </div>
              <div style={styles.bootstrapStat}>
                <span style={styles.statLabel}>CI Width:</span>
                <span style={styles.statValue}>{ciWidth.toFixed(4)}</span>
              </div>
            </div>

            {/* Visual CI representation */}
            <div style={styles.ciVisualization}>
              <div style={styles.ciBar}>
                <div
                  style={{
                    ...styles.ciRange,
                    left: `${(bootstrap.ci_lower / (bootstrap.ci_upper * 1.2)) * 100}%`,
                    width: `${(ciWidth / (bootstrap.ci_upper * 1.2)) * 100}%`
                  }}
                ></div>
                <div
                  style={{
                    ...styles.ciMean,
                    left: `${(bootstrap.mean / (bootstrap.ci_upper * 1.2)) * 100}%`
                  }}
                ></div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const BestMatchDisplay = ({ bestMatchResults, bootstrapResults }) => {
  if (!bestMatchResults) return null;

  const {
    best_matches_1to2,
    best_matches_2to1,
    reciprocal_matches,
    coverage_1to2,
    coverage_2to1,
    high_divergence_1to2,
    high_divergence_2to1,
    num_reciprocal
  } = bestMatchResults;

  return (
    <div style={styles.bestMatchContainer}>
      <h3 style={styles.metricTitle}>
        🎯 Best-Match Analysis
        <HelpTooltip text="Identifies the closest matching topics between datasets. Reciprocal matches indicate stable topics that persist across corpus sizes." />
      </h3>

      {/* Coverage Statistics */}
      <div style={styles.coverageStats}>
        <h4 style={styles.subsectionTitle}>Coverage Statistics</h4>
        <div style={styles.statsGrid}>
          <div style={styles.statCard}>
            <div style={styles.statCardValue}>{(coverage_1to2 * 100).toFixed(1)}%</div>
            <div style={styles.statCardLabel}>Dataset 1→2 Coverage@0.3</div>
            <div style={styles.statCardDescription}>
              Topics in Dataset 1 with good matches in Dataset 2 (TVD &lt; 0.3)
            </div>
          </div>

          <div style={styles.statCard}>
            <div style={styles.statCardValue}>{(coverage_2to1 * 100).toFixed(1)}%</div>
            <div style={styles.statCardLabel}>Dataset 2→1 Coverage@0.3</div>
            <div style={styles.statCardDescription}>
              Topics in Dataset 2 with good matches in Dataset 1 (TVD &lt; 0.3)
            </div>
          </div>

          <div style={styles.statCard}>
            <div style={styles.statCardValue}>{num_reciprocal}</div>
            <div style={styles.statCardLabel}>Reciprocal Matches</div>
            <div style={styles.statCardDescription}>
              Topic pairs that mutually select each other as best matches
            </div>
          </div>

          <div style={{...styles.statCard, backgroundColor: high_divergence_1to2 > 0.3 ? '#5D2A2A' : '#2A2A2A'}}>
            <div style={styles.statCardValue}>{(high_divergence_1to2 * 100).toFixed(1)}%</div>
            <div style={styles.statCardLabel}>High Divergence 1→2</div>
            <div style={styles.statCardDescription}>
              Topics with poor matches (TVD &gt; 0.7) indicating instability
            </div>
          </div>

          <div style={{...styles.statCard, backgroundColor: high_divergence_2to1 > 0.3 ? '#5D2A2A' : '#2A2A2A'}}>
            <div style={styles.statCardValue}>{(high_divergence_2to1 * 100).toFixed(1)}%</div>
            <div style={styles.statCardLabel}>High Divergence 2→1</div>
            <div style={styles.statCardDescription}>
              Topics with poor matches (TVD &gt; 0.7) indicating instability
            </div>
          </div>
        </div>

        {bootstrapResults && (
          <div style={styles.bootstrapCIBox}>
            <h5>Bootstrap Confidence Intervals (1000 samples):</h5>
            <div style={styles.bootstrapResults}>
              <div>
                Mean TVD 1→2: {bootstrapResults.mean_tvd_1to2.mean.toFixed(3)}
                [{bootstrapResults.mean_tvd_1to2.ci_lower.toFixed(3)}, {bootstrapResults.mean_tvd_1to2.ci_upper.toFixed(3)}]
              </div>
              <div>
                Mean TVD 2→1: {bootstrapResults.mean_tvd_2to1.mean.toFixed(3)}
                [{bootstrapResults.mean_tvd_2to1.ci_lower.toFixed(3)}, {bootstrapResults.mean_tvd_2to1.ci_upper.toFixed(3)}]
              </div>
              <div>
                Reciprocal Count: {bootstrapResults.reciprocal_count.mean.toFixed(1)}
                [{bootstrapResults.reciprocal_count.ci_lower.toFixed(0)}, {bootstrapResults.reciprocal_count.ci_upper.toFixed(0)}]
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Reciprocal Matches Table */}
      {reciprocal_matches && reciprocal_matches.length > 0 && (
        <div style={styles.reciprocalMatchesSection}>
          <h4 style={styles.subsectionTitle}>
            🔄 Reciprocal Matches (Stable Topics)
          </h4>
          <div style={styles.matchesTable}>
            {reciprocal_matches.map((match, idx) => (
              <div key={idx} style={styles.matchRow}>
                <div style={styles.matchTopic}>
                  <div style={styles.matchTopicLabel}>Dataset 1 - {match.topic1_name}</div>
                </div>
                <div style={styles.matchArrowContainer}>
                  <div style={styles.matchArrowBidirectional}>⟷</div>
                  <div style={{...styles.matchTvdBadge, backgroundColor: match.tvd < 0.3 ? '#4CAF50' : match.tvd < 0.7 ? '#FFC107' : '#F44336'}}>
                    TVD: {match.tvd.toFixed(3)}
                  </div>
                </div>
                <div style={styles.matchTopic}>
                  <div style={styles.matchTopicLabel}>Dataset 2 - {match.topic2_name}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Best Matches 1→2 */}
      <details style={styles.detailsSection}>
        <summary style={styles.detailsSummary}>
          📤 Best Matches: Dataset 1 → Dataset 2 ({best_matches_1to2.length} topics)
        </summary>
        <div style={styles.matchesTable}>
          {best_matches_1to2.map((match, idx) => (
            <div key={idx} style={styles.matchRow}>
              <div style={styles.matchTopic}>
                <div style={styles.matchTopicLabel}>{match.topic1_name}</div>
              </div>
              <div style={styles.matchArrowContainer}>
                <div style={styles.matchArrowUnidirectional}>→</div>
                <div style={{...styles.matchTvdBadge, backgroundColor: match.tvd < 0.3 ? '#4CAF50' : match.tvd < 0.7 ? '#FFC107' : '#F44336'}}>
                  TVD: {match.tvd.toFixed(3)}
                </div>
              </div>
              <div style={styles.matchTopic}>
                <div style={styles.matchTopicLabel}>{match.topic2_name}</div>
              </div>
            </div>
          ))}
        </div>
      </details>

      {/* Best Matches 2→1 */}
      <details style={styles.detailsSection}>
        <summary style={styles.detailsSummary}>
          📥 Best Matches: Dataset 2 → Dataset 1 ({best_matches_2to1.length} topics)
        </summary>
        <div style={styles.matchesTable}>
          {best_matches_2to1.map((match, idx) => (
            <div key={idx} style={styles.matchRow}>
              <div style={styles.matchTopic}>
                <div style={styles.matchTopicLabel}>{match.topic2_name}</div>
              </div>
              <div style={styles.matchArrowContainer}>
                <div style={styles.matchArrowUnidirectional}>→</div>
                <div style={{...styles.matchTvdBadge, backgroundColor: match.tvd < 0.3 ? '#4CAF50' : match.tvd < 0.7 ? '#FFC107' : '#F44336'}}>
                  TVD: {match.tvd.toFixed(3)}
                </div>
              </div>
              <div style={styles.matchTopic}>
                <div style={styles.matchTopicLabel}>{match.topic1_name}</div>
              </div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
};

const SimilarityHeatmap = ({ matrixData, topics1, topics2 }) => {
  if (!matrixData || !topics1 || !topics2) return null;

  const getSimilarityColor = (similarity) => {
    if (similarity >= 80) return '#4CAF50'; // Green - Very High
    if (similarity >= 60) return '#8BC34A'; // Light Green - High
    if (similarity >= 40) return '#FFC107'; // Yellow - Medium
    if (similarity >= 20) return '#FF9800'; // Orange - Low
    return '#F44336'; // Red - Very Low
  };

  return (
    <div style={styles.heatmapContainer}>
      <h3 style={styles.heatmapTitle}>📊 Topic Similarity Matrix (TVD)</h3>
      <p style={styles.heatmapDescription}>
        Pairwise Total Variation Distance between all topic pairs
      </p>

      {/* Simple color legend */}
      <div style={styles.legendContainer}>
        <div style={styles.legend}>
          <div style={styles.legendItem}>
            <div style={{...styles.legendColor, backgroundColor: '#4CAF50'}}></div>
            <span>Very High (80-100%)</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{...styles.legendColor, backgroundColor: '#8BC34A'}}></div>
            <span>High (60-80%)</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{...styles.legendColor, backgroundColor: '#FFC107'}}></div>
            <span>Medium (40-60%)</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{...styles.legendColor, backgroundColor: '#FF9800'}}></div>
            <span>Low (20-40%)</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{...styles.legendColor, backgroundColor: '#F44336'}}></div>
            <span>Very Low (0-20%)</span>
          </div>
        </div>
      </div>

      {/* Simple table matrix */}
      <div style={styles.matrixContainer}>
        {/* Header row */}
        <div style={styles.matrixHeader}>
          <div style={styles.cornerCell}></div>
          {topics2.map((topic, j) => (
            <div key={j} style={styles.headerCell}>
              <div style={{fontSize: '0.7rem', fontWeight: 'bold', marginBottom: '2px'}}>
                Dataset 2 - Topic {topic.id}
              </div>
              <div style={{fontSize: '0.6rem', color: '#B0B0B0'}}>
                {topic.name.split(',').slice(0, 2).join(', ')}
              </div>
            </div>
          ))}
        </div>

        {/* Data rows */}
        {matrixData.map((row, i) => (
          <div key={i} style={styles.matrixRow}>
            <div style={styles.rowHeaderCell}>
              <div style={{fontSize: '0.7rem', fontWeight: 'bold', marginBottom: '2px'}}>
                Dataset 1 - Topic {topics1[i].id}
              </div>
              <div style={{fontSize: '0.6rem', color: '#B0B0B0'}}>
                {topics1[i].name.split(',').slice(0, 2).join(', ')}
              </div>
            </div>
            {row.map((cell, j) => (
              <div
                key={j}
                style={{
                  ...styles.matrixCell,
                  backgroundColor: getSimilarityColor(cell.similarity_percent),
                  color: cell.similarity_percent > 50 ? '#FFF' : '#000'
                }}
                title={`Similarity: ${cell.similarity_percent.toFixed(1)}%\nTVD Distance: ${cell.tvd_distance.toFixed(3)}`}
              >
                {cell.similarity_percent.toFixed(0)}%
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

  return (
    <div style={styles.container}>
      {showWelcome && <WelcomeOverlay onDismiss={() => setShowWelcome(false)} />}
      <h1 style={styles.header}>📚 PDF Topic Analysis</h1>
      <FileUpload
        getRootProps={getRootProps}
        getInputProps={getInputProps}
        file={file}
      />
      <ModelSelection
        modelType={modelType}
        setModelType={setModelType}
      />
      {modelType === "LDA" ? (
        <LDASettings
          numTopics={numTopics}
          setNumTopics={setNumTopics}
          numWords={numWords}
          setNumWords={setNumWords}
        />
      ) : (
        <HDPSettings
          numWords={numWords}
          setNumWords={setNumWords}
        />
      )}
      <BibliographySettings
        includeBibliography={includeBibliography}
        setIncludeBibliography={setIncludeBibliography}
        includeDecadeAnalysis={includeDecadeAnalysis}
        setIncludeDecadeAnalysis={setIncludeDecadeAnalysis}
        stopwords={stopwords}
        setStopwords={setStopwords}
        numTopPapers={numTopPapers}
        setNumTopPapers={setNumTopPapers}
      />
      <button onClick={handleAnalyze} style={styles.button} disabled={loading}>
        {loading ? `Analyzing with ${modelType}...` : `🔍 Run ${modelType} Analysis`}
      </button>
      <div style={styles.resultsContainer}>
        <div style={styles.headerRow}>
          <h3 style={styles.sectionHeader}>📊 {modelType} Analysis Results</h3>
          <button
            style={styles.infoButton}
            onClick={() => setShowExplanation(!showExplanation)}
          >
            ℹ
          </button>
        </div>

        {showExplanation && explanationContent}
        {loading ? (
          <p>Loading analysis...</p>
        ) : results && results.topics && results.topics.length > 0 ? (
          <>
            <div>
              <h4>Topics:</h4>
              {topics && topics.length > 0 ? (
                <div>
                  <div style={styles.dropdownContainer}>
                    <label htmlFor="topicDropdown" style={styles.dropdownLabel}>
                      Select a Topic:
                    </label>
                    <select
                      id="topicDropdown"
                      value={selectedTopic ? selectedTopic.Topic : ""}
                      onChange={handleTopicSelect}
                      style={styles.topicDropdown} // Apply topicDropdown styles here
                    >
                      <option value="" style={styles.dropdownOption}>
                        -- Select a Topic --
                      </option>
                      {topics.map((topic, index) => (
                        <option
                          key={index}
                          value={topic.Topic}
                          style={styles.dropdownOption}
                        >
                          {topic.Topic}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedTopic && (
                    <div>
                      <h4>{selectedTopic.Topic}</h4>
                      <p>
                        <strong>Words:</strong>{" "}
                        {selectedTopic.Words || "No words available."}
                      </p>
                    </div>
                  )}
                  {selectedTopic && results?.top_papers && (
                    <div style={styles.topPapersContainer}>
                      <h4>Top Papers for {selectedTopic.Topic}</h4>
                      <div style={styles.papersList}>
                        {(() => {
                          const topicIndex = parseInt(selectedTopic.Topic.split(" ")[1]) - 1;
                          const papers = results.top_papers[topicIndex];
                          return papers?.map((paper, index) => (
                          <div key={index} style={styles.paperItem}>
                            <div style={styles.paperInfo}>
                              <div style={styles.paperTitle}>{paper.title}</div>
                              <div style={styles.paperDetails}>
                                <span style={styles.paperAuthor}>{paper.author}</span>
                                <span style={styles.paperYear}>{paper.year}</span>
                                {/* Always show PubMed link if ID exists */}
                                {paper.pubmed_id ? (
                                  <a
                                    href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pubmed_id}/`}
                                    style={styles.pubmedLink}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    title="Open in PubMed"
                                  >
                                    PubMed
                                  </a>
                                ) : (
                                  <span style={styles.pubmedLink}>No PubMed ID</span>
                                )}
                              </div>
                            </div>
                            <div style={styles.loadingBarContainer}>
                              <div
                                style={{
                                  ...styles.loadingBar,
                                  width: `${paper.loading_factor * 100}%`,
                                }}
                              >
                                <span style={styles.loadingValue}>
                                  {(paper.loading_factor * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                          ));
                        })()}
                      </div>
                    </div>
                  )}
                  {/* Render Chart for the Selected Topic */}
                  {chartBase64 && selectedTopic && (
                    <div style={styles.selectedTopicChart}>
                      <img
                        src={`data:image/png;base64,${
                          chartBase64[selectedTopic.Topic]
                        }`}
                        alt={`${selectedTopic.Topic} distribution chart`}
                        style={styles.responsiveChartImage}
                      />
                    </div>
                  )}
                  {includeDecadeAnalysis && decadeChartBase64 && (
                    <div>
                      <h4>📅 Decade Analysis Chart</h4>
                      <div style={styles.selectedTopicChart}>
                        <img
                          src={`data:image/png;base64,${decadeChartBase64}`}
                          alt="Decade Analysis Chart"
                          style={styles.responsiveChartImage}
                        />
                      </div>
                    </div>
                  )}
                  
                  {/* HDP Advanced Visualizations */}
                  {modelType === "HDP" && chartBase64 && (
                    <div style={styles.hdpAdvancedCharts}>
                      <h3 style={styles.sectionHeader}>🎨 Advanced HDP Visualizations</h3>
                      
                      {/* Multi-Panel Overview */}
                      {chartBase64["HDP_Multi_Panel_Overview"] && (
                        <div style={styles.advancedChartSection}>
                          <h4>📊 Complete Topic Overview</h4>
                          <div style={styles.selectedTopicChart}>
                            <img
                              src={`data:image/png;base64,${chartBase64["HDP_Multi_Panel_Overview"]}`}
                              alt="HDP Multi-Panel Topic Overview"
                              style={styles.responsiveChartImage}
                            />
                          </div>
                        </div>
                      )}
                      
                      {/* Quality Heatmap */}
                      {chartBase64["HDP_Quality_Heatmap"] && (
                        <div style={styles.advancedChartSection}>
                          <h4>🔥 Topic Quality Heatmap</h4>
                          <p style={styles.chartDescription}>
                            Shows topic strength metrics: average strength, maximum strength, and document coverage.
                          </p>
                          <div style={styles.selectedTopicChart}>
                            <img
                              src={`data:image/png;base64,${chartBase64["HDP_Quality_Heatmap"]}`}
                              alt="HDP Topic Quality Heatmap"
                              style={styles.responsiveChartImage}
                            />
                          </div>
                        </div>
                      )}
                      
                      {/* Word Clouds */}
                      {chartBase64["HDP_Word_Clouds"] && (
                        <div style={styles.advancedChartSection}>
                          <h4>☁️ Top Topic Word Clouds</h4>
                          <p style={styles.chartDescription}>
                            Visual word clouds for the strongest topics discovered by HDP.
                          </p>
                          <div style={styles.selectedTopicChart}>
                            <img
                              src={`data:image/png;base64,${chartBase64["HDP_Word_Clouds"]}`}
                              alt="HDP Word Clouds"
                              style={styles.responsiveChartImage}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <p>No topics available.</p>
              )}
            </div>
          </>
        ) : (
          <p>Results will appear here after analysis.</p>
        )}
        {results && (
          <div style={styles.exportButtons}>
            <button style={styles.exportButton} onClick={handleExportCSV}>
              📥 Export Papers (CSV)
            </button>
            <button style={styles.exportButton} onClick={handleExportTopicDistributions}>
              📊 Export Topic Distributions (CSV)
            </button>
            <button style={styles.exportButton} onClick={handleExportTopicPrevalence}>
              📈 Export Topic Prevalence (CSV)
            </button>
            <button style={styles.exportButton} onClick={handleExportCharts}>
              🖼️ Export Charts (ZIP)
            </button>
          </div>
        )}
        
        {/* Topic Comparison Section */}
        <div style={styles.comparisonSection}>
          <h2 style={styles.sectionHeader}>📊 Advanced Topic Similarity Analysis</h2>
          <p style={styles.comparisonDescription}>
            Compare topic distributions from two different analyses using advanced metrics:
            Total Variation Distance (TVD), Optimal Transport (OT) distance, and best-match analysis.
            Upload the "Topic Distributions CSV" files from two separate analyses.
            Optionally include "Topic Prevalence CSV" files for weighted analysis.
          </p>

          <div style={styles.fileInputContainer}>
            <div style={styles.fileInputGroup}>
              <label style={styles.fileLabel}>
                Dataset 1 - Topic Distributions (Required):
                <HelpTooltip text="Upload the topic-word distributions CSV exported from your first analysis" />
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setComparisonFile1(e.target.files[0])}
                style={styles.fileInput}
              />
              {comparisonFile1 && <span style={styles.fileName}>{comparisonFile1.name}</span>}
            </div>

            <div style={styles.fileInputGroup}>
              <label style={styles.fileLabel}>
                Dataset 1 - Topic Prevalence (Optional):
                <HelpTooltip text="Upload the topic prevalence CSV for weighted OT distance calculation. If not provided, uniform weights will be used." />
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setPrevalenceFile1(e.target.files[0])}
                style={styles.fileInput}
              />
              {prevalenceFile1 && <span style={styles.fileName}>{prevalenceFile1.name}</span>}
            </div>
          </div>

          <div style={styles.fileInputContainer}>
            <div style={styles.fileInputGroup}>
              <label style={styles.fileLabel}>
                Dataset 2 - Topic Distributions (Required):
                <HelpTooltip text="Upload the topic-word distributions CSV exported from your second analysis" />
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setComparisonFile2(e.target.files[0])}
                style={styles.fileInput}
              />
              {comparisonFile2 && <span style={styles.fileName}>{comparisonFile2.name}</span>}
            </div>

            <div style={styles.fileInputGroup}>
              <label style={styles.fileLabel}>
                Dataset 2 - Topic Prevalence (Optional):
                <HelpTooltip text="Upload the topic prevalence CSV for weighted OT distance calculation. If not provided, uniform weights will be used." />
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setPrevalenceFile2(e.target.files[0])}
                style={styles.fileInput}
              />
              {prevalenceFile2 && <span style={styles.fileName}>{prevalenceFile2.name}</span>}
            </div>
          </div>
          
          <button 
            onClick={handleCompareTopics} 
            style={styles.button} 
            disabled={isComparing || !comparisonFile1 || !comparisonFile2}
          >
            {isComparing ? "Calculating Similarities..." : "📊 Calculate Topic Similarities"}
          </button>
          
          {comparisonResults && (
            <div style={styles.comparisonResults}>
              {/* Summary Statistics */}
              <div style={styles.comparisonSummary}>
                <h4>Analysis Summary:</h4>
                <ul>
                  <li>Dataset 1: {comparisonResults.num_topics1} topics</li>
                  <li>Dataset 2: {comparisonResults.num_topics2} topics</li>
                  <li>Combined vocabulary: {comparisonResults.vocabulary_size} unique words</li>
                  <li>Prevalence data: {comparisonResults.has_prevalence ? '✅ Provided' : '⚠️ Not provided (using uniform weights)'}</li>
                </ul>
              </div>

              {/* Optimal Transport Distance Display */}
              {comparisonResults.optimal_transport && (
                <OptimalTransportDisplay
                  otResults={comparisonResults.optimal_transport}
                  hasPrevalence={comparisonResults.has_prevalence}
                />
              )}

              {/* Best Match Analysis Display */}
              {comparisonResults.best_match_analysis && (
                <BestMatchDisplay
                  bestMatchResults={comparisonResults.best_match_analysis}
                  bootstrapResults={comparisonResults.best_match_bootstrap}
                />
              )}

              {/* Traditional TVD Heatmap */}
              <SimilarityHeatmap
                matrixData={comparisonResults.matrix_data}
                topics1={comparisonResults.topics1}
                topics2={comparisonResults.topics2}
              />
            </div>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

// Dark Mode Styles
const styles = {
  container: {
    padding: "30px",
    marginTop: "40px",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#121212",
    borderRadius: "8px",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.5)",
    maxWidth: "700px",
    margin: "auto",
    textAlign: "center",
    color: "#E0E0E0",
  },
  header: {
    color: "#FFF",
    fontSize: "36px",
    marginBottom: "20px",
  },
  dropzone: {
    border: "2px dashed #FFFFFF",
    padding: "30px",
    marginBottom: "20px",
    backgroundColor: "#1F1F1F",
    cursor: "pointer",
    borderRadius: "8px",
    color: "#E0E0E0",
  },
  settingsContainer: {
    marginBottom: "20px",
  },
  modelSelectionContainer: {
    marginBottom: "20px",
    textAlign: "left",
    backgroundColor: "#1F1F1F",
    padding: "20px",
    borderRadius: "8px",
    border: "1px solid #333",
  },
  modelOptions: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
    marginTop: "15px",
  },
  modelOption: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "12px",
    backgroundColor: "#2A2A2A",
    borderRadius: "6px",
    cursor: "pointer",
    transition: "background-color 0.3s ease",
    "&:hover": {
      backgroundColor: "#333",
    },
  },
  radioInput: {
    width: "18px",
    height: "18px",
    accentColor: "#6200EE",
  },
  modelLabel: {
    color: "#E0E0E0",
    fontSize: "1rem",
    fontWeight: "500",
    flex: 1,
  },
  setting: {
    marginBottom: "15px",
    textAlign: "left",
  },
  hdpInfoBox: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    marginBottom: "20px",
    border: "1px solid #444",
  },
  hdpInfoTitle: {
    color: "#6200EE",
    fontSize: "1.1rem",
    marginBottom: "10px",
    marginTop: "0",
  },
  hdpInfoText: {
    color: "#B0B0B0",
    fontSize: "0.9rem",
    lineHeight: "1.4",
    margin: "0 0 15px 0",
  },
  hdpInfoList: {
    color: "#E0E0E0",
    fontSize: "0.9rem",
    lineHeight: "1.6",
    margin: "15px 0",
    paddingLeft: "20px",
  },
  hdpAdvancedCharts: {
    marginTop: "40px",
    padding: "20px",
    backgroundColor: "#1A1A1A",
    borderRadius: "8px",
    border: "1px solid #333",
  },
  advancedChartSection: {
    marginBottom: "30px",
  },
  chartDescription: {
    color: "#B0B0B0",
    fontSize: "0.9rem",
    marginBottom: "15px",
    fontStyle: "italic",
  },
  input: {
    backgroundColor: "#333",
    color: "#E0E0E0",
    border: "1px solid #444",
    padding: "5px",
    borderRadius: "4px",
    width: "100px",
  },
  button: {
    backgroundColor: "#6200EE",
    color: "#FFF",
    padding: "10px 20px",
    fontSize: "16px",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  resultsContainer: {
    marginTop: "30px",
    textAlign: "left",
  },
  results: {
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    textAlign: "left",
    color: "#FFF",
  },
  stopwordsInput: {
    backgroundColor: "#333",
    color: "#E0E0E0",
    border: "1px solid #444",
    padding: "5px",
    borderRadius: "4px",
    width: "100%",
  },
  bibliographyContainer: {
    textAlign: "left",
    color: "#E0E0E0",
  },
  sectionHeader: {
    fontSize: "18px",
    marginBottom: "10px",
  },
  checkboxLabel: {
    fontSize: "16px",
    color: "#E0E0E0",
  },
  checkbox: {
    marginRight: "10px",
  },
  dropdownContainer: {
    margin: "10px 0",
    position: "relative",
    maxWidth: "200px",
    width: "100%",
    backgroundColor: "#121212", // Black background
  },
  dropdownLabel: {
    display: "block",
    marginBottom: "8px",
    color: "#FFFFFF", // White text
    fontSize: "0.95rem",
    fontWeight: "500",
    backgroundColor: "#121212", // Black background
  },
  topicDropdown: {
    width: "100%",
    padding: "12px 16px",
    fontSize: "0.85rem",
    backgroundColor: "#1A1A1A", // Dark grey background
    border: "1px solid #333", // Grey border
    borderRadius: "8px",
    color: "#FFFFFF", // White text
    appearance: "none",
    transition: "all 0.3s ease",
    cursor: "pointer",
    "&:hover": {
      borderColor: "#555", // Lighter grey border on hover
      backgroundColor: "#2A2A2A", // Slightly lighter grey background
    },
    "&:focus": {
      outline: "none",
      borderColor: "#4CAF50", // Green border on focus
      boxShadow: "0 0 0 2px rgba(76, 175, 80, 0.2)", // Green focus shadow
    },
  },
  dropdownOption: {
    backgroundColor: "#1F1F1F", // Dark grey option background
    color: "#FFFFFF", // White text
    padding: "12px",
    fontSize: "0.85rem",
    "&:hover": {
      backgroundColor: "#4CAF50", // Green background on hover
      color: "#FFFFFF", // White text on hover
    },
  },
  dropdownArrow: {
    position: "absolute",
    right: "15px",
    top: "50%",
    transform: "translateY(-50%)",
    pointerEvents: "none",
    color: "#AAAAAA", // Light grey arrow
    fontSize: "1.2rem",
  },

  selectedTopicChart: {
    width: "95%",
    maxWidth: "1000px",
    margin: "20px 0",
    padding: "20px",
    backgroundColor: "#fff",
    borderRadius: "8px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
  },
  responsiveChartImage: {
    width: "100%",
    height: "auto",
    maxHeight: "600px",
    objectFit: "contain",
    border: "1px solid #eee",
    borderRadius: "4px",
  },
  chartTitle: {
    color: "#333",
    marginBottom: "15px",
    fontSize: "1.25rem",
    fontWeight: "500",
  },
  chartContainer: {
    width: "100%",
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "20px 0",
  },
  exportButtons: {
    margin: "20px 0",
    display: "flex",
    gap: "10px",
  },
  exportButton: {
    padding: "10px 20px",
    backgroundColor: "#6200EE",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "14px",
    "&:hover": {
      backgroundColor: "#45a049",
    },
  },
  chartImage: {
    width: "100%",
    height: "auto",
    maxHeight: "80vh",
    objectFit: "contain",
  },
  decadeChartContainer: {
    width: "50%",
    marginTop: "30px",
    padding: "20px",
    backgroundColor: "#f8f9fa",
    borderRadius: "8px",
  },
  headerRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  },
  infoButton: {
    background: "none",
    border: "none",
    color: "#FFFFFF",
    fontSize: "1.1rem",
    cursor: "pointer",
    padding: "5px",
    borderRadius: "50%",
    width: "28px",
    height: "28px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    "&:hover": {
      backgroundColor: "#f0f0f0",
    },
  },
  explanationBox: {
    position: "relative",
    backgroundColor: "#f8f9fa",
    borderRadius: "6px",
    padding: "15px",
    margin: "10px 0",
    border: "1px solid #dee2e6",
    fontSize: "0.9rem",
  },
  topPapersContainer: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#2A2A2A',
    borderRadius: '8px',
  },
  topPapersTable: {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: '10px',
  },
  topPapersTableTh: {
    backgroundColor: '#4CAF50',
    color: 'white',
    padding: '10px',
    textAlign: 'left',
  },
  topPapersTableTd: {
    padding: '10px',
    borderBottom: '1px solid #444',
  },
  explanationContent: {
    marginRight: "20px",
  },
  explanationText: {
    margin: "0 0 10px 0",
    color: "#495057",
    lineHeight: "1.4",
  },
  explanationList: {
    margin: "0",
    paddingLeft: "20px",
    color: "#6c757d",
  },
  closeButton: {
    position: "absolute",
    top: "5px",
    right: "5px",
    background: "none",
    border: "none",
    color: "#6c757d",
    fontSize: "1.3rem",
    cursor: "pointer",
    padding: "2px 5px",
    "&:hover": {
      color: "#495057",
    },
  },
  wordScores: {
    marginTop: "10px",
    padding: "10px",
    backgroundColor: "#2A2A2A",
    borderRadius: "6px",
  },
  scoreRow: {
    display: "flex",
    justifyContent: "space-between",
    margin: "5px 0",
  },
  wordLabel: {
    flex: 2,
    textAlign: "left",
  },
  percentage: {
    flex: 1,
    textAlign: "right",
    color: "#4CAF50",
    margin: "0 10px",
  },
  rawScore: {
    flex: 1,
    textAlign: "right",
    color: "#9E9E9E",
    fontSize: "0.8em",
  },
  topicSelection: {
    margin: "20px 0",
  },
  topicDetail: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    margin: "20px 0",
  },
  decadeAnalysis: {
    marginTop: "40px",
    padding: "20px",
    backgroundColor: "#2A2A2A",
    borderRadius: "8px",
  },
  statsContainer: {
    backgroundColor: "#2A2A2A",
    padding: "15px",
    borderRadius: "8px",
    margin: "20px 0",
  },
  statItem: {
    display: "flex",
    justifyContent: "space-between",
    margin: "8px 0",
    padding: "8px",
    backgroundColor: "#333",
    borderRadius: "4px",
  },
  statLabel: {
    color: "#E0E0E0",
    fontWeight: "500",
  },
  statValue: {
    color: "#4CAF50",
    fontWeight: "600",
  },
  tooltipContainer: {
    position: 'relative',
    display: 'inline-block',
    marginLeft: '8px',
  },
  tooltipIcon: {
    background: '#6200EE',
    color: 'white',
    border: 'none',
    borderRadius: '50%',
    width: '20px',
    height: '20px',
    cursor: 'help',
    fontSize: '14px',
    lineHeight: '20px',
    textAlign: 'center',
    verticalAlign: 'middle',
    marginRight: '20px',
    marginLeft: '0px',
    marginBottom: '5px',
  },
  tooltipContent: {
    position: 'absolute',
    bottom: '100%',
    left: '50%',
    transform: 'translateX(-50%)',
    backgroundColor: '#2A2A2A',
    color: '#E0E0E0',
    padding: '10px',
    borderRadius: '6px',
    width: '250px',
    fontSize: '0.9rem',
    boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    zIndex: 1000,
  },
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.9)',
    zIndex: 2000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlayContent: {
    backgroundColor: '#1F1F1F',
    padding: '2rem',
    borderRadius: '12px',
    maxWidth: '600px',
    textAlign: 'center',
  },
  overlayList: {
    textAlign: 'left',
    margin: '1.5rem 0',
    lineHeight: '1.6',
    paddingLeft: '1.5rem',
  },
  overlayButton: {
    backgroundColor: '#6200EE',
    color: 'white',
    padding: '12px 24px',
    fontSize: '1.1rem',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  papersList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  paperItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
    padding: '10px',
    backgroundColor: '#1F1F1F',
    borderRadius: '6px',
  },
  paperInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  paperTitle: {
    fontSize: '0.95rem',
    fontWeight: '500',
    color: '#E0E0E0',
  },
  paperDetails: {
    display: 'flex',
    gap: '10px',
    fontSize: '0.85rem',
    color: '#888',
  },
  loadingBarContainer: {
    width: '100%',
    height: '20px',
    backgroundColor: '#333',
    borderRadius: '10px',
    overflow: 'hidden',
    position: 'relative',
  },
  loadingBar: {
    height: '100%',
    backgroundColor: '#6200EE',
    borderRadius: '10px',
    transition: 'width 0.3s ease',
  },
  loadingValue: {
    position: 'absolute',
    right: '8px',
    top: '50%',
    transform: 'translateY(-50%)',
    fontSize: '0.8rem',
    color: '#FFF',
  },
  pubmedLink: {
    color: "#4CAF50",
    textDecoration: "none",
    marginLeft: "8px",
    fontSize: "0.8rem",
    border: "1px solid #4CAF50",
    borderRadius: "4px",
    padding: "2px 6px",
    "&:hover": {
      backgroundColor: "#4CAF5020",
    },
  },
  comparisonSection: {
    marginTop: "50px",
    padding: "30px",
    backgroundColor: "#1A1A1A",
    borderRadius: "8px",
    border: "1px solid #333",
  },
  comparisonDescription: {
    color: "#B0B0B0",
    fontSize: "0.9rem",
    marginBottom: "20px",
    lineHeight: "1.5",
  },
  fileInputContainer: {
    display: "flex",
    gap: "20px",
    marginBottom: "20px",
    flexWrap: "wrap",
  },
  fileInputGroup: {
    flex: "1",
    minWidth: "300px",
  },
  fileLabel: {
    display: "block",
    color: "#E0E0E0",
    marginBottom: "8px",
    fontSize: "0.9rem",
    fontWeight: "500",
  },
  fileInput: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#2A2A2A",
    border: "1px solid #444",
    borderRadius: "4px",
    color: "#E0E0E0",
    fontSize: "0.9rem",
  },
  fileName: {
    display: "block",
    color: "#4CAF50",
    fontSize: "0.8rem",
    marginTop: "5px",
    fontStyle: "italic",
  },
  comparisonResults: {
    marginTop: "30px",
  },
  comparisonSummary: {
    marginTop: "20px",
    padding: "15px",
    backgroundColor: "#2A2A2A",
    borderRadius: "6px",
    color: "#E0E0E0",
  },
  heatmapContainer: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    marginBottom: "20px",
  },
  heatmapTitle: {
    color: "#FFF",
    fontSize: "1.5rem",
    marginBottom: "8px",
    textAlign: "center",
  },
  heatmapDescription: {
    color: "#B0B0B0",
    fontSize: "0.9rem",
    marginBottom: "15px",
    fontStyle: "italic",
    textAlign: "center",
  },
  matrixContainer: {
    display: "inline-block",
    border: "1px solid #444",
    borderRadius: "4px",
    overflow: "hidden",
  },
  matrixHeader: {
    display: "flex",
  },
  matrixRow: {
    display: "flex",
  },
  cornerCell: {
    width: "100px",
    height: "60px",
    backgroundColor: "#333",
    border: "1px solid #444",
  },
  headerCell: {
    width: "80px",
    height: "60px",
    backgroundColor: "#333",
    border: "1px solid #444",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#E0E0E0",
    fontSize: "0.8rem",
    textAlign: "center",
    fontWeight: "500",
  },
  rowHeaderCell: {
    width: "100px",
    height: "50px",
    backgroundColor: "#333",
    border: "1px solid #444",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#E0E0E0",
    fontSize: "0.8rem",
    textAlign: "center",
    fontWeight: "500",
  },
  matrixCell: {
    width: "80px",
    height: "50px",
    border: "1px solid #444",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "0.75rem",
    fontWeight: "600",
    cursor: "pointer",
  },
  legendContainer: {
    marginBottom: "20px",
  },
  legend: {
    display: "flex",
    flexWrap: "wrap",
    gap: "15px",
    justifyContent: "center",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "0.85rem",
    color: "#E0E0E0",
  },
  legendColor: {
    width: "20px",
    height: "15px",
    borderRadius: "3px",
    border: "1px solid #666",
  },
  // New styles for Optimal Transport Display
  otContainer: {
    backgroundColor: "#2A2A2A",
    padding: "25px",
    borderRadius: "8px",
    marginBottom: "25px",
    border: "1px solid #444",
  },
  metricTitle: {
    color: "#FFF",
    fontSize: "1.3rem",
    marginBottom: "15px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  otMainMetric: {
    display: "flex",
    gap: "30px",
    flexWrap: "wrap",
  },
  otDistanceBox: {
    flex: "0 0 250px",
    backgroundColor: "#333",
    padding: "20px",
    borderRadius: "8px",
    textAlign: "center",
    border: "2px solid #6200EE",
  },
  otDistanceValue: {
    fontSize: "2.5rem",
    fontWeight: "bold",
    color: "#6200EE",
    marginBottom: "10px",
  },
  otDistanceLabel: {
    fontSize: "1rem",
    color: "#E0E0E0",
    marginBottom: "5px",
  },
  otWarning: {
    fontSize: "0.75rem",
    color: "#FFC107",
    marginTop: "10px",
    fontStyle: "italic",
  },
  otBootstrapInfo: {
    flex: "1",
    backgroundColor: "#333",
    padding: "20px",
    borderRadius: "8px",
  },
  bootstrapTitle: {
    color: "#E0E0E0",
    fontSize: "1rem",
    marginBottom: "15px",
    marginTop: "0",
  },
  bootstrapStats: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap: "10px",
    marginBottom: "20px",
  },
  bootstrapStat: {
    backgroundColor: "#2A2A2A",
    padding: "10px",
    borderRadius: "6px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  ciVisualization: {
    marginTop: "20px",
  },
  ciBar: {
    position: "relative",
    height: "30px",
    backgroundColor: "#1A1A1A",
    borderRadius: "4px",
    overflow: "hidden",
  },
  ciRange: {
    position: "absolute",
    height: "100%",
    backgroundColor: "#6200EE",
    opacity: "0.3",
  },
  ciMean: {
    position: "absolute",
    width: "3px",
    height: "100%",
    backgroundColor: "#6200EE",
    boxShadow: "0 0 5px #6200EE",
  },
  // New styles for Best Match Display
  bestMatchContainer: {
    backgroundColor: "#2A2A2A",
    padding: "25px",
    borderRadius: "8px",
    marginBottom: "25px",
    border: "1px solid #444",
  },
  subsectionTitle: {
    color: "#E0E0E0",
    fontSize: "1.1rem",
    marginBottom: "15px",
    marginTop: "0",
  },
  coverageStats: {
    marginBottom: "25px",
  },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "15px",
    marginBottom: "20px",
  },
  statCard: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    border: "1px solid #444",
    textAlign: "center",
  },
  statCardValue: {
    fontSize: "2rem",
    fontWeight: "bold",
    color: "#4CAF50",
    marginBottom: "8px",
  },
  statCardLabel: {
    fontSize: "0.9rem",
    color: "#E0E0E0",
    marginBottom: "8px",
    fontWeight: "500",
  },
  statCardDescription: {
    fontSize: "0.75rem",
    color: "#B0B0B0",
    lineHeight: "1.4",
  },
  bootstrapCIBox: {
    backgroundColor: "#333",
    padding: "15px",
    borderRadius: "6px",
    marginTop: "15px",
  },
  bootstrapResults: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    marginTop: "10px",
    color: "#E0E0E0",
    fontSize: "0.9rem",
  },
  reciprocalMatchesSection: {
    marginTop: "25px",
    marginBottom: "25px",
  },
  matchesTable: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  matchRow: {
    display: "flex",
    alignItems: "center",
    gap: "15px",
    backgroundColor: "#333",
    padding: "15px",
    borderRadius: "6px",
    border: "1px solid #444",
  },
  matchTopic: {
    flex: "1",
    minWidth: "0",
  },
  matchTopicLabel: {
    fontSize: "0.9rem",
    color: "#E0E0E0",
    wordWrap: "break-word",
  },
  matchArrowContainer: {
    flex: "0 0 auto",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "5px",
  },
  matchArrowBidirectional: {
    fontSize: "1.5rem",
    color: "#4CAF50",
    fontWeight: "bold",
  },
  matchArrowUnidirectional: {
    fontSize: "1.5rem",
    color: "#2196F3",
    fontWeight: "bold",
  },
  matchTvdBadge: {
    padding: "4px 10px",
    borderRadius: "4px",
    fontSize: "0.75rem",
    fontWeight: "600",
    color: "#FFF",
  },
  detailsSection: {
    backgroundColor: "#333",
    padding: "15px",
    borderRadius: "6px",
    marginTop: "15px",
    border: "1px solid #444",
  },
  detailsSummary: {
    fontSize: "1rem",
    color: "#E0E0E0",
    cursor: "pointer",
    fontWeight: "500",
    padding: "5px",
    userSelect: "none",
  },
  modernMatrixContainer: {
    display: "inline-block",
    backgroundColor: "#1A1A1A",
    borderRadius: "8px",
    overflow: "hidden",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
  },
  topicLabelsRow: {
    display: "flex",
    backgroundColor: "#333",
  },
  cornerLabel: {
    width: "200px",
    height: "80px",
    backgroundColor: "#444",
  },
  topicLabel: {
    width: "140px",
    height: "80px",
    backgroundColor: "#333",
    borderLeft: "1px solid #555",
    padding: "8px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    textAlign: "center",
  },
  topicNumber: {
    fontSize: "0.75rem",
    fontWeight: "600",
    color: "#FFF",
    marginBottom: "4px",
  },
  topicWords: {
    fontSize: "0.7rem",
    color: "#B0B0B0",
    lineHeight: "1.2",
    wordBreak: "break-word",
  },
  matrixDataRow: {
    display: "flex",
    borderTop: "1px solid #555",
  },
  leftTopicLabel: {
    width: "200px",
    height: "70px",
    backgroundColor: "#333",
    padding: "8px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    textAlign: "center",
  },
  modernMatrixCell: {
    width: "140px",
    height: "70px",
    borderLeft: "1px solid #555",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    transition: "transform 0.2s ease, box-shadow 0.2s ease",
    "&:hover": {
      transform: "scale(1.05)",
      boxShadow: "0 2px 8px rgba(255,255,255,0.2)",
    },
  },
  cellTvd: {
    fontSize: "1rem",
    fontWeight: "700",
    marginBottom: "2px",
  },
  cellSimilarity: {
    fontSize: "0.8rem",
    fontWeight: "500",
    opacity: "0.9",
  },
  // New responsive styles
  responsiveHeatmapContainer: {
    width: "100%",
    backgroundColor: "#1A1A1A",
    borderRadius: "12px",
    padding: "25px",
    margin: "20px 0",
  },
  heatmapHeader: {
    textAlign: "center",
    marginBottom: "25px",
  },
  heatmapSubtitle: {
    color: "#B0B0B0",
    fontSize: "1rem",
    fontStyle: "italic",
  },
  interpretationGuide: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    marginBottom: "25px",
  },
  guideGrid: {
    display: "flex",
    gap: "20px",
    flexWrap: "wrap",
    justifyContent: "space-around",
    marginTop: "15px",
  },
  guideItem: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    minWidth: "200px",
  },
  guideColor: {
    width: "25px",
    height: "25px",
    borderRadius: "6px",
    border: "2px solid #555",
  },
  scrollableMatrix: {
    overflowX: "auto",
    overflowY: "visible",
    marginBottom: "25px",
  },
  cardBasedMatrix: {
    display: "flex",
    flexDirection: "column",
    gap: "25px",
    minWidth: "800px",
  },
  topicComparisonSection: {
    display: "flex",
    alignItems: "center",
    gap: "20px",
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "10px",
    border: "1px solid #444",
  },
  sourceTopicCard: {
    flex: "0 0 250px",
    backgroundColor: "#333",
    borderRadius: "8px",
    padding: "15px",
  },
  topicCardHeader: {
    borderBottom: "1px solid #555",
    paddingBottom: "10px",
    marginBottom: "10px",
  },
  topicCardContent: {
    color: "#E0E0E0",
  },
  comparisonArrow: {
    fontSize: "2rem",
    color: "#6200EE",
    flex: "0 0 40px",
    textAlign: "center",
  },
  similarityGrid: {
    display: "flex",
    gap: "15px",
    flex: "1",
    overflowX: "auto",
  },
  similarityCard: {
    flex: "0 0 220px",
    backgroundColor: "#333",
    borderRadius: "8px",
    padding: "15px",
    border: "1px solid #555",
  },
  targetTopicHeader: {
    borderBottom: "1px solid #555",
    paddingBottom: "10px",
    marginBottom: "15px",
  },
  targetTopicWords: {
    fontSize: "0.85rem",
    color: "#B0B0B0",
    margin: "5px 0 0 0",
  },
  similarityMetrics: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "15px",
  },
  similarityBadge: {
    borderRadius: "8px",
    padding: "10px",
    textAlign: "center",
    minWidth: "60px",
    fontWeight: "bold",
  },
  metricValue: {
    fontSize: "1.2rem",
    fontWeight: "700",
  },
  metricLabel: {
    fontSize: "0.7rem",
    opacity: "0.9",
  },
  tvdInfo: {
    textAlign: "right",
  },
  tvdValue: {
    fontSize: "0.9rem",
    color: "#E0E0E0",
    fontWeight: "600",
  },
  tvdLabel: {
    fontSize: "0.75rem",
    color: "#B0B0B0",
    marginTop: "2px",
  },
  summaryCard: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    border: "1px solid #444",
  },
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "15px",
    marginTop: "15px",
  },
  summaryItem: {
    padding: "10px",
    backgroundColor: "#333",
    borderRadius: "6px",
    color: "#E0E0E0",
  },
  // Network visualization styles
  networkContainer: {
    width: "100%",
    backgroundColor: "#1A1A1A",
    borderRadius: "12px",
    padding: "25px",
    margin: "20px 0",
  },
  networkHeader: {
    textAlign: "center",
    marginBottom: "25px",
  },
  networkSubtitle: {
    color: "#B0B0B0",
    fontSize: "1rem",
    fontStyle: "italic",
    marginTop: "8px",
  },
  networkVisualization: {
    position: "relative",
    width: "100%",
    minHeight: "400px",
    backgroundColor: "#0F0F0F",
    borderRadius: "8px",
    margin: "25px 0",
    overflow: "hidden",
  },
  connectionSvg: {
    position: "absolute",
    top: 0,
    left: 0,
    zIndex: 1,
  },
  leftTopics: {
    position: "absolute",
    left: "20px",
    top: "20px",
    width: "200px",
    zIndex: 2,
  },
  rightTopics: {
    position: "absolute",
    right: "20px",
    top: "20px",
    width: "200px",
    zIndex: 2,
  },
  corpusLabel: {
    backgroundColor: "#333",
    padding: "12px",
    borderRadius: "8px",
    textAlign: "center",
    marginBottom: "20px",
    border: "2px solid #555",
  },
  topicNode: {
    position: "absolute",
    width: "200px",
    height: "60px",
    borderRadius: "8px",
    padding: "8px",
    color: "white",
    boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
    border: "2px solid rgba(255,255,255,0.1)",
    backdropFilter: "blur(10px)",
  },
  topicNodeHeader: {
    fontSize: "0.9rem",
    fontWeight: "600",
    marginBottom: "4px",
    opacity: "0.9",
  },
  topicNodeWords: {
    fontSize: "0.75rem",
    lineHeight: "1.2",
    opacity: "0.8",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  networkLegend: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
    marginBottom: "20px",
  },
  legendItems: {
    display: "flex",
    gap: "25px",
    flexWrap: "wrap",
    marginTop: "15px",
  },
  legendLine: {
    width: "40px",
    borderRadius: "2px",
    marginRight: "12px",
  },
  matchesSummary: {
    backgroundColor: "#2A2A2A",
    padding: "20px",
    borderRadius: "8px",
  },
  matchesList: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
    marginTop: "15px",
  },
  matchItem: {
    backgroundColor: "#333",
    padding: "15px",
    borderRadius: "6px",
    border: "1px solid #555",
  },
  matchConnection: {
    display: "flex",
    alignItems: "center",
    gap: "15px",
  },
  matchSource: {
    flex: "1",
    color: "#2196F3",
    fontSize: "0.9rem",
    fontWeight: "500",
  },
  matchArrow: {
    flex: "0 0 120px",
    position: "relative",
    height: "20px",
    backgroundColor: "#555",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  matchLine: {
    height: "100%",
    backgroundColor: "#4CAF50",
    borderRadius: "10px",
    transition: "width 0.3s ease",
  },
  matchPercent: {
    position: "absolute",
    fontSize: "0.75rem",
    fontWeight: "600",
    color: "white",
  },
  matchTarget: {
    flex: "1",
    color: "#9C27B0",
    fontSize: "0.9rem",
    fontWeight: "500",
    textAlign: "right",
  },
};

export default App;
