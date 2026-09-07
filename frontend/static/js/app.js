"use strict";

const API_ENDPOINTS = Object.freeze({
  health: "/health",
  prediction: "/api/v1/predict",
  feedback: "/api/v1/feedback",
});

const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png"]);
const MAXIMUM_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_CLASSES = new Set(["normal", "scratch", "dent", "rust"]);

const state = {
  selectedFile: null,
  previewUrl: null,
  currentPrediction: null,
  apiReady: false,
  requestInProgress: false,
  feedbackSubmitted: false,
};

const elements = {};

function cacheElements() {
  const identifiers = [
    "api-status",
    "prediction-form",
    "drop-zone",
    "image-input",
    "preview-panel",
    "preview-image",
    "selected-file-name",
    "selected-file-details",
    "remove-image-button",
    "analyse-button",
    "loading-area",
    "error-message",
    "result-placeholder",
    "result-content",
    "result-status-badge",
    "predicted-class",
    "most-likely-class",
    "prediction-confidence",
    "confidence-track",
    "confidence-bar",
    "manual-review-message",
    "knowledge-source-label",
    "repair-steps",
    "repair-empty",
    "inventory-data-source",
    "inventory-last-updated",
    "parts-table-body",
    "parts-empty",
    "feedback-yes",
    "feedback-no",
    "feedback-details",
    "corrected-class",
    "feedback-comment",
    "feedback-status",
    "poc-disclaimer",
    "analyse-another-button",
  ];

  identifiers.forEach((identifier) => {
    const element = document.getElementById(identifier);
    if (!element) {
      throw new Error(`Required interface element is missing: ${identifier}`);
    }
    elements[identifier] = element;
  });
}

function formatFileSize(sizeBytes) {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function titleCase(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function validateSelectedFile(file) {
  if (!file) {
    return "Select a JPEG or PNG image before analysis.";
  }
  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    return "Unsupported file type. Select a JPEG or PNG image.";
  }
  if (file.size <= 0) {
    return "The selected image is empty.";
  }
  if (file.size > MAXIMUM_FILE_SIZE_BYTES) {
    return "The selected image exceeds the 10 MB upload limit.";
  }
  return null;
}

function updateAnalyseButton() {
  elements["analyse-button"].disabled =
    !state.selectedFile || !state.apiReady || state.requestInProgress;
}

function clearPreviewUrl() {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
}

function handleImageSelection(event) {
  const file = event.target.files?.[0] ?? null;
  const validationError = validateSelectedFile(file);
  if (validationError) {
    resetSelectedImage();
    renderError(validationError);
    return;
  }

  clearPreviewUrl();
  clearError();
  state.selectedFile = file;
  state.previewUrl = URL.createObjectURL(file);
  elements["preview-image"].src = state.previewUrl;
  elements["selected-file-name"].textContent = file.name;
  elements["selected-file-details"].textContent = `${formatFileSize(file.size)} · ${file.type}`;
  elements["preview-panel"].hidden = false;
  elements["drop-zone"].hidden = true;
  updateAnalyseButton();
}

function resetSelectedImage() {
  clearPreviewUrl();
  state.selectedFile = null;
  elements["image-input"].value = "";
  elements["preview-image"].removeAttribute("src");
  elements["selected-file-name"].textContent = "";
  elements["selected-file-details"].textContent = "";
  elements["preview-panel"].hidden = true;
  elements["drop-zone"].hidden = false;
  updateAnalyseButton();
}

function setLoading(isLoading) {
  state.requestInProgress = isLoading;
  elements["loading-area"].hidden = !isLoading;
  elements["prediction-form"].setAttribute("aria-busy", String(isLoading));
  elements["image-input"].disabled = isLoading;
  elements["remove-image-button"].disabled = isLoading;
  updateAnalyseButton();
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };
  if (!response.ok) {
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : "The server could not complete the request.";
    throw new Error(detail);
  }
  return body;
}

async function submitPrediction(event) {
  event.preventDefault();
  clearError();

  const validationError = validateSelectedFile(state.selectedFile);
  if (validationError) {
    renderError(validationError);
    return;
  }
  if (!state.apiReady) {
    renderError("The classification API or trained model is not ready.");
    return;
  }

  const formData = new FormData();
  formData.append("file", state.selectedFile, state.selectedFile.name);
  setLoading(true);

  try {
    const response = await fetch(API_ENDPOINTS.prediction, {
      method: "POST",
      body: formData,
    });
    const result = await parseResponse(response);
    state.currentPrediction = result;
    renderPrediction(result);
  } catch (error) {
    renderError(error instanceof Error ? error.message : "Image analysis failed.");
  } finally {
    setLoading(false);
  }
}

function renderPrediction(result) {
  const confidence = Number(result.confidence);
  const confidencePercent = Math.round(confidence * 1000) / 10;
  const requiresReview = Boolean(result.requires_manual_review);
  const feedbackApplied = Boolean(result.feedback_applied);

  elements["result-placeholder"].hidden = true;
  elements["result-content"].hidden = false;
  elements["predicted-class"].textContent = result.display_name;
  elements["most-likely-class"].textContent = feedbackApplied
    ? `Model result: ${titleCase(result.most_likely_class)}`
    : requiresReview
    ? `Model's closest class: ${titleCase(result.most_likely_class)}`
    : result.filename;
  elements["prediction-confidence"].textContent = feedbackApplied
    ? `Model was ${confidencePercent.toFixed(1)}% confident before correction`
    : `${confidencePercent.toFixed(1)}%`;
  elements["confidence-bar"].style.width = `${Math.min(100, Math.max(0, confidencePercent))}%`;
  elements["confidence-track"].setAttribute(
    "aria-valuenow",
    String(Math.round(confidencePercent)),
  );

  elements["result-status-badge"].textContent = feedbackApplied
    ? "Saved correction applied"
    : requiresReview
      ? "Manual review required"
      : "Classification complete";
  elements["result-status-badge"].className = requiresReview
    ? "result-status-badge review"
    : "result-status-badge complete";
  elements["result-content"].classList.toggle("requires-review", requiresReview);

  elements["manual-review-message"].hidden = !requiresReview;
  elements["manual-review-message"].textContent = requiresReview
    ? result.message
    : "";
  renderKnowledgeSource(result.knowledge_source);
  elements["inventory-last-updated"].textContent =
    result.inventory_last_updated || "Not provided";
  elements["poc-disclaimer"].textContent = result.poc_disclaimer;

  renderRepairSteps(result.repair_steps);
  renderParts(result.parts);
  resetFeedbackControls();
  elements["result-content"].scrollIntoView({ behavior: "smooth", block: "start" });
}

function knowledgeSourceLabel(value) {
  return value === "neo4j"
    ? "Neo4j knowledge graph"
    : "Static catalog fallback";
}

function renderKnowledgeSource(value) {
  const source = value === "neo4j" ? "neo4j" : "static_catalog_fallback";
  const label = knowledgeSourceLabel(source);
  elements["knowledge-source-label"].textContent = label;
  elements["knowledge-source-label"].className =
    `source-label ${source === "neo4j" ? "graph" : "fallback"}`;
  elements["inventory-data-source"].textContent = label;
}

function renderRepairSteps(steps) {
  elements["repair-steps"].replaceChildren();
  const safeSteps = Array.isArray(steps) ? steps : [];
  elements["repair-empty"].hidden = safeSteps.length !== 0;
  elements["repair-empty"].textContent = state.currentPrediction?.requires_manual_review
    ? "Repair guidance is withheld until the image is reviewed manually."
    : "No repair action is required for this classification.";

  safeSteps.forEach((step) => {
    const item = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = String(step);
    item.append(text);
    elements["repair-steps"].append(item);
  });
}

function availabilityLabel(value) {
  const normalized = String(value).trim().toLowerCase();
  const labels = {
    in_stock: "In stock",
    low_stock: "Low stock",
    out_of_stock: "Out of stock",
  };
  return labels[normalized] ?? titleCase(normalized);
}

function renderParts(parts) {
  elements["parts-table-body"].replaceChildren();
  const safeParts = Array.isArray(parts) ? parts : [];
  elements["parts-empty"].hidden = safeParts.length !== 0;
  elements["parts-empty"].textContent = state.currentPrediction?.requires_manual_review
    ? "Parts are not shown for an unknown classification."
    : "No replacement part is required for this classification.";

  safeParts.forEach((part) => {
    const row = document.createElement("tr");
    const partCell = document.createElement("td");
    const priceCell = document.createElement("td");
    const stockCell = document.createElement("td");
    const availabilityCell = document.createElement("td");
    const partName = document.createElement("strong");
    const partId = document.createElement("small");
    const availability = document.createElement("span");

    partName.textContent = String(part.part_name);
    partId.textContent = String(part.part_id);
    partCell.append(partName, partId);

    try {
      priceCell.textContent = new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: String(part.currency),
        maximumFractionDigits: 2,
      }).format(Number(part.mock_price));
    } catch {
      priceCell.textContent = `${part.currency} ${part.mock_price}`;
    }
    stockCell.textContent = `${part.fake_stock_quantity} units`;
    availability.textContent = availabilityLabel(part.availability_status);
    availability.className = `availability ${String(part.availability_status).toLowerCase()}`;
    availabilityCell.append(availability);

    [partCell, priceCell, stockCell, availabilityCell].forEach((cell) => row.append(cell));
    elements["parts-table-body"].append(row);
  });
}

function resetFeedbackControls() {
  state.feedbackSubmitted = false;
  elements["feedback-yes"].disabled = false;
  elements["feedback-no"].disabled = false;
  elements["corrected-class"].disabled = false;
  elements["feedback-comment"].disabled = false;
  elements["corrected-class"].value = "";
  elements["feedback-comment"].value = "";
  elements["feedback-status"].textContent = "";
  elements["feedback-status"].className = "feedback-status";
}

async function submitFeedback(isCorrect) {
  if (!state.currentPrediction || state.feedbackSubmitted) {
    return;
  }
  const correctedClass = elements["corrected-class"].value || null;
  if (!isCorrect && !correctedClass) {
    elements["feedback-status"].textContent =
      "Select the correct class before marking the result incorrect.";
    elements["feedback-status"].className = "feedback-status error";
    return;
  }
  if (correctedClass && !SUPPORTED_CLASSES.has(correctedClass)) {
    elements["feedback-status"].textContent = "Select a supported corrected class.";
    elements["feedback-status"].className = "feedback-status error";
    return;
  }

  elements["feedback-yes"].disabled = true;
  elements["feedback-no"].disabled = true;
  elements["feedback-status"].textContent = "Saving feedback…";
  elements["feedback-status"].className = "feedback-status";

  const payload = {
    prediction_id: state.currentPrediction.prediction_id,
    image_filename: state.currentPrediction.filename,
    image_sha256: state.currentPrediction.image_sha256,
    predicted_class: state.currentPrediction.predicted_class,
    confidence: state.currentPrediction.confidence,
    is_correct: isCorrect,
    corrected_class: isCorrect ? null : correctedClass,
    comment: elements["feedback-comment"].value.trim() || null,
  };

  try {
    const response = await fetch(API_ENDPOINTS.feedback, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await parseResponse(response);
    state.feedbackSubmitted = true;
    elements["corrected-class"].disabled = true;
    elements["feedback-comment"].disabled = true;
    elements["feedback-status"].textContent = result.message;
    elements["feedback-status"].className = "feedback-status success";
  } catch (error) {
    elements["feedback-yes"].disabled = false;
    elements["feedback-no"].disabled = false;
    elements["feedback-status"].textContent =
      error instanceof Error ? error.message : "Feedback could not be saved.";
    elements["feedback-status"].className = "feedback-status error";
  }
}

function renderError(message) {
  elements["error-message"].textContent = message;
  elements["error-message"].hidden = false;
}

function clearError() {
  elements["error-message"].textContent = "";
  elements["error-message"].hidden = true;
}

function resetInterface() {
  resetSelectedImage();
  clearError();
  state.currentPrediction = null;
  elements["result-content"].hidden = true;
  elements["result-placeholder"].hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function checkApiHealth() {
  elements["api-status"].className = "api-status checking";
  try {
    const response = await fetch(API_ENDPOINTS.health, {
      headers: { Accept: "application/json" },
    });
    const health = await parseResponse(response);
    state.apiReady = Boolean(health.model_loaded && health.catalog_loaded);
    if (state.apiReady) {
      const graphConnected = Boolean(health.neo4j_connected);
      elements["api-status"].className = graphConnected
        ? "api-status ready"
        : "api-status degraded";
      elements["api-status"].lastChild.textContent = graphConnected
        ? " API ready · Neo4j"
        : " API ready · static fallback";
    } else {
      elements["api-status"].className = "api-status degraded";
      elements["api-status"].lastChild.textContent = " Model not trained";
    }
  } catch {
    state.apiReady = false;
    elements["api-status"].className = "api-status offline";
    elements["api-status"].lastChild.textContent = " API offline";
  }
  updateAnalyseButton();
}

function bindEvents() {
  elements["image-input"].addEventListener("change", handleImageSelection);
  elements["prediction-form"].addEventListener("submit", submitPrediction);
  elements["remove-image-button"].addEventListener("click", resetSelectedImage);
  elements["feedback-yes"].addEventListener("click", () => submitFeedback(true));
  elements["feedback-no"].addEventListener("click", () => submitFeedback(false));
  elements["analyse-another-button"].addEventListener("click", resetInterface);

  ["dragenter", "dragover"].forEach((eventName) => {
    elements["drop-zone"].addEventListener(eventName, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.add("drag-active");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements["drop-zone"].addEventListener(eventName, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.remove("drag-active");
    });
  });
  elements["drop-zone"].addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) {
      return;
    }
    const transfer = new DataTransfer();
    transfer.items.add(file);
    elements["image-input"].files = transfer.files;
    elements["image-input"].dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function initializeApplication() {
  cacheElements();
  bindEvents();
  updateAnalyseButton();
  checkApiHealth();
}

document.addEventListener("DOMContentLoaded", initializeApplication);
