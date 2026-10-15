
document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. DASHBOARD DATA
    // ==========================================

    const dashboardStats = {
        totalScans: 0,
        categories: new Set(),
        highRisk: 0,
        ecoActions: 0,
        risks: { Low: 0, Medium: 0, High: 0 },
        recentScans: []
    };

    const ecoTips = [
        "Plant native trees to support local wildlife and biodiversity.",
        "Save water by fixing leaks and avoiding unnecessary water use.",
        "Reduce plastic waste by carrying reusable bags and bottles.",
        "Protect natural habitats and avoid disturbing wild animals.",
        "Separate recyclable waste from general household waste.",
        "Use electricity wisely by switching off unused lights.",
        "Keep rivers, parks, and forests free from litter.",
        "Choose walking or cycling for short journeys when practical."
    ];

    const $ = (id) => document.getElementById(id);

    function updateDashboard() {
        if ($("totalScans")) {
            $("totalScans").textContent = dashboardStats.totalScans;
        }

        if ($("natureCategories")) {
            $("natureCategories").textContent =
                dashboardStats.categories.size;
        }

        if ($("highRiskCount")) {
            $("highRiskCount").textContent = dashboardStats.highRisk;
        }

        if ($("ecoActions")) {
            $("ecoActions").textContent = dashboardStats.ecoActions;
        }

        const totalPredictions = Object.values(dashboardStats.risks)
            .reduce((sum, count) => sum + count, 0);

        ["Low", "Medium", "High"].forEach((risk) => {
            const key = risk.toLowerCase();
            const label = $(`${key}RiskLabel`);
            const bar = $(`${key}RiskBar`);
            const count = dashboardStats.risks[risk];

            if (label) label.textContent = count;

            if (bar) {
                const percentage = totalPredictions
                    ? (count / totalPredictions) * 100
                    : 0;

                bar.style.width = `${percentage}%`;
                bar.setAttribute("aria-valuenow", percentage);
            }
        });

        renderRecentScans();
    }

    function renderRecentScans() {
        const container = $("recentScans");
        if (!container) return;

        container.replaceChildren();

        if (!dashboardStats.recentScans.length) {
            const empty = document.createElement("p");
            empty.className = "dashboard-empty";
            empty.textContent =
                "No scans yet. Visit the Nature Scanner to analyze an image.";
            container.appendChild(empty);
            return;
        }

        dashboardStats.recentScans.forEach((scan) => {
            const item = document.createElement("div");
            item.className = "recent-scan-item";

            const image = document.createElement("img");
            image.src = scan.imageUrl || "";
            image.alt = "Recently scanned image";
            image.loading = "lazy";

            const details = document.createElement("div");
            const heading = document.createElement("h4");
            heading.textContent = scan.label || "Unknown classification";

            const description = document.createElement("p");
            const confidence = scan.confidence ?? "N/A";
            description.textContent =
                `${scan.category || "Unknown"} · ${confidence}` +
                (confidence === "N/A" ? "" : "% confidence");

            details.append(heading, description);
            item.append(image, details);
            container.appendChild(item);
        });
    }

    // Load permanent statistics saved by Flask and SQLite.
    async function loadPersistentDashboard() {
        try {
            const response = await fetch("/dashboard-stats");
            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "Could not load dashboard statistics."
                );
            }

            dashboardStats.totalScans = Number(data.totalScans) || 0;
            dashboardStats.highRisk = Number(data.highRiskCount) || 0;
            dashboardStats.ecoActions = Number(data.ecoActions) || 0;

            // Display the saved number of distinct categories.
            dashboardStats.categories = new Set(
                Array.from(
                    { length: Number(data.natureCategories) || 0 },
                    (_, index) => `saved-category-${index}`
                )
            );

            dashboardStats.risks = {
                Low: Number(data.risks?.Low) || 0,
                Medium: Number(data.risks?.Medium) || 0,
                High: Number(data.risks?.High) || 0
            };

            dashboardStats.recentScans =
                (data.recentScans || []).map((scan) => ({
                    label: scan.label,
                    category: scan.nature_category || scan.label,
                    confidence: scan.confidence ?? "N/A",
                    imageUrl: scan.image_url
                }));

            updateDashboard();
        } catch (error) {
            console.error("Dashboard loading error:", error);
        }
    }

    // Initial display, then load database records.
    updateDashboard();
    loadPersistentDashboard();

    // ==========================================
    // 2. NATURE RISK PREDICTOR - SLIDERS
    // ==========================================

    const sliderNames = [
        "vegetation",
        "rainfall",
        "soil_quality",
        "human_activity"
    ];

    sliderNames.forEach((name) => {
        const slider = $(name);
        const output = $(`${name}Value`);

        if (slider && output) {
            output.textContent = slider.value;

            slider.addEventListener("input", () => {
                output.textContent = slider.value;
            });
        }
    });

    // ==========================================
    // 3. NATURE RISK PREDICTOR - SUBMISSION
    // ==========================================

    const predictorForm = $("predictorForm");
    const predictionResult = $("predictionResult");

    if (predictorForm && predictionResult) {
        predictorForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const button = predictorForm.querySelector(
                'button[type="submit"]'
            );
            const originalText = button?.textContent || "Analyze";

            if (button) {
                button.disabled = true;
                button.textContent = "Analyzing...";
            }

            const payload = {};

            sliderNames.forEach((name) => {
                const slider = $(name);
                if (slider) payload[name] = Number(slider.value);
            });

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Prediction failed.");
                }

                predictionResult.replaceChildren();
                predictionResult.classList.remove(
                    "risk-low", "risk-medium", "risk-high"
                );

                const risk = String(data.risk || "Medium").toLowerCase();
                predictionResult.classList.add(`risk-${risk}`);

                const icon = document.createElement("div");
                icon.className = "result-icon";
                icon.textContent = "🌿";

                const label = document.createElement("span");
                label.className = "card-label";
                label.textContent = "YOUR ANALYSIS";

                const heading = document.createElement("h3");
                heading.textContent = `${data.risk} Risk`;

                const recommendation = document.createElement("p");
                recommendation.textContent = data.recommendation || "";

                const divider = document.createElement("div");
                divider.className = "result-divider";

                const note = document.createElement("span");
                note.className = "result-footnote";
                note.textContent = data.note || "";

                predictionResult.append(
                    icon, label, heading, recommendation, divider, note
                );

                // Flask saves the prediction to SQLite.
                await loadPersistentDashboard();

            } catch (error) {
                predictionResult.replaceChildren();

                const heading = document.createElement("h3");
                heading.textContent = "Unable to analyze";

                const message = document.createElement("p");
                message.textContent = error.message;

                predictionResult.append(heading, message);
                console.error("Prediction error:", error);
            } finally {
                if (button) {
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }
        });
    }

    // ==========================================
    // 4. ECOBOT CHATBOT
    // ==========================================

    const chatForm = $("chatForm");
    const chatInput = $("chatInput");
    const chatMessages = $("chatMessages");
    const chatSend = $("chatSend");

    function addChatMessage(message, type) {
        if (!chatMessages) return;

        const element = document.createElement("div");
        element.className = `chat-message ${type}-message`;
        element.textContent = message;

        chatMessages.appendChild(element);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    if (chatForm && chatInput && chatMessages && chatSend) {
        chatForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const message = chatInput.value.trim();
            if (!message) return;

            addChatMessage(message, "user");
            chatInput.value = "";
            chatSend.disabled = true;
            chatSend.textContent = "Thinking...";

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Unable to contact EcoBot.");
                }

                addChatMessage(
                    data.reply || "EcoBot did not return a response.",
                    "bot"
                );
            } catch (error) {
                addChatMessage(
                    "Sorry, I couldn't connect to EcoBot. Check that Flask is running.",
                    "bot"
                );
                console.error("EcoBot error:", error);
            } finally {
                chatSend.disabled = false;
                chatSend.textContent = "Send";
                chatInput.focus();
            }
        });

        document.querySelectorAll(".chat-suggestion").forEach((button) => {
            button.addEventListener("click", () => {
                chatInput.value = button.textContent.trim();
                chatForm.requestSubmit();
            });
        });
    }

    // ==========================================
    // 5. NATURE SCANNER - IMAGE PREVIEW
    // ==========================================

    const scannerForm = $("scannerForm");
    const imageInput = $("natureImage");
    const previewArea = $("scannerPreview");
    const previewImage = $("previewImage");
    const scannerResult = $("scannerResult");
    const scanButton = $("scanButton");

    let previewUrl = null;

    if (
        scannerForm && imageInput && previewArea &&
        previewImage && scannerResult && scanButton
    ) {
        imageInput.addEventListener("change", () => {
            const file = imageInput.files[0];

            scannerResult.hidden = true;

            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
                previewUrl = null;
            }

            if (!file) {
                previewArea.hidden = true;
                previewImage.removeAttribute("src");
                return;
            }

            const allowedTypes = [
                "image/jpeg", "image/png", "image/webp"
            ];

            if (
                !allowedTypes.includes(file.type) ||
                file.size > 5 * 1024 * 1024
            ) {
                alert("Choose a JPG, PNG, or WEBP image under 5 MB.");
                imageInput.value = "";
                previewArea.hidden = true;
                previewImage.removeAttribute("src");
                return;
            }

            previewUrl = URL.createObjectURL(file);
            previewImage.src = previewUrl;
            previewArea.hidden = false;
        });

        // ==========================================
        // 6. NATURE SCANNER - AI CLASSIFICATION
        // ==========================================

        scannerForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const file = imageInput.files[0];

            if (!file) {
                alert("Please select an image first.");
                return;
            }

            if (file.size > 5 * 1024 * 1024) {
                alert("Please choose an image smaller than 5 MB.");
                return;
            }

            if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
                alert("Choose a JPG, PNG, or WEBP image.");
                return;
            }

            const formData = new FormData();
            formData.append("image", file);

            scanButton.disabled = true;
            scanButton.textContent = "🔍 Analyzing image...";
            scannerResult.hidden = false;
            scannerResult.replaceChildren();

            const loading = document.createElement("p");
            loading.textContent =
                "Please wait while the AI analyzes your image. The first scan may take longer.";
            scannerResult.appendChild(loading);

            try {
                const response = await fetch("/scan", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Image analysis failed.");
                }

                scannerResult.replaceChildren();

                const heading = document.createElement("h3");
                heading.textContent = "🌿 AI Prediction";

                const image = document.createElement("img");
                image.src = data.image_url || "";
                image.alt = "Image analyzed by Nature Scanner";

                const label = document.createElement("p");
                label.textContent =
                    `Predicted category: ${data.label || "Unknown"}`;

                const category = document.createElement("p");
                category.textContent =
                    `Nature category: ${data.nature_category || data.label || "Not available"}`;

                const confidence = document.createElement("p");
                confidence.textContent =
                    `Model confidence: ${data.confidence ?? "N/A"}` +
                    (data.confidence != null ? "%" : "");

                const confidenceMessage = document.createElement("p");
                confidenceMessage.textContent =
                    data.confidence_message ||
                    "Model confidence is an estimate, not a guarantee.";

                const tipHeading = document.createElement("h4");
                tipHeading.textContent = "🌱 Conservation Tip";

                const tip = document.createElement("p");
                tip.textContent =
                    data.tip || "Help protect and preserve natural habitats.";

                const alternativesHeading = document.createElement("h4");
                alternativesHeading.textContent = "Other possible classifications";

                const alternativesList = document.createElement("ul");

                (Array.isArray(data.alternatives) ? data.alternatives : [])
                    .forEach((item) => {
                        const listItem = document.createElement("li");

                        if (typeof item === "string") {
                            listItem.textContent = item;
                        } else if (item && typeof item === "object") {
                            const name = item.label || item.name || "Alternative";
                            const score = item.confidence ?? item.score;

                            listItem.textContent = score == null
                                ? name
                                : `${name}: ${score}%`;
                        }

                        alternativesList.appendChild(listItem);
                    });

                const note = document.createElement("p");
                note.className = "scanner-note";
                note.textContent = data.note || "";

                scannerResult.append(
                    heading, image, label, category, confidence,
                    confidenceMessage, tipHeading, tip
                );

                if (alternativesList.children.length) {
                    scannerResult.append(alternativesHeading, alternativesList);
                }

                scannerResult.appendChild(note);

                // Reload statistics saved by Flask/SQLite.
                if (data.image_url) {
                    await loadPersistentDashboard();
                }
            } catch (error) {
                scannerResult.replaceChildren();

                const heading = document.createElement("h3");
                heading.textContent = "Unable to scan image";

                const message = document.createElement("p");
                message.className = "scanner-error";
                message.textContent = error.message;

                scannerResult.append(heading, message);
                console.error("Nature Scanner error:", error);
            } finally {
                scanButton.disabled = false;
                scanButton.textContent = "🌱 Analyze Image";
            }
        });
    }

    // ==========================================
    // 7. DASHBOARD - ECO TIPS
    // ==========================================

    const newEcoTipButton = $("newEcoTip");
    const dashboardEcoTip = $("dashboardEcoTip");

    if (newEcoTipButton && dashboardEcoTip) {
        let previousTipIndex = -1;

        newEcoTipButton.addEventListener("click", async () => {
            let nextTipIndex;

            do {
                nextTipIndex = Math.floor(Math.random() * ecoTips.length);
            } while (
                ecoTips.length > 1 && nextTipIndex === previousTipIndex
            );

            previousTipIndex = nextTipIndex;
            dashboardEcoTip.textContent = ecoTips[nextTipIndex];

            newEcoTipButton.disabled = true;

            try {
                const response = await fetch("/dashboard-eco-action", {
                    method: "POST"
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.error || "Could not save eco action."
                    );
                }

                await loadPersistentDashboard();
            } catch (error) {
                console.error("Eco action saving error:", error);
                dashboardEcoTip.textContent +=
                    " (The action could not be saved.)";
            } finally {
                newEcoTipButton.disabled = false;
            }
        });
    }

    // ==========================================
    // 8. FEATURE-CARD NAVIGATION
    // ==========================================

    document.querySelectorAll("[data-target]").forEach((button) => {
        button.addEventListener("click", () => {
            const sectionIds = {
                predictor: "predictor",
                scanner: "scanner",
                ecobot: "ecobot",
                dashboard: "dashboard"
            };

            const targetSection = $(
                sectionIds[button.dataset.target]
            );

            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    });

    // ==========================================
    // 9. SCROLL REVEAL ANIMATIONS
    // ==========================================

    const elements = document.querySelectorAll(
        ".section-heading, .feature-card, " +
        ".predictor-form, .prediction-result, " +
        ".ecobot-card, .scanner-card, .mission-strip, " +
        ".bottom-banner, .dashboard-stat, .dashboard-panel"
    );

    if (
        "IntersectionObserver" in window &&
        !window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
        elements.forEach((element) => {
            element.classList.add("reveal-on-scroll");
        });

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    obs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.12 });

        elements.forEach((element) => observer.observe(element));
    }
});
