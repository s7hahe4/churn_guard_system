/**
 * ChurnGuard AI - Frontend Controller & Data Visualization
 */

// Chart instance references hoisted at module scope
let riskChartInstance = null;
let contractChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 0. Theme Manager (Light & Dark Mode)
    // -------------------------------------------------------------
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    let savedTheme = 'dark';
    try {
        savedTheme = localStorage.getItem('churnguard_theme') || 'dark';
    } catch(e) {}

    function applyTheme(theme) {
        const isLight = (theme === 'light');
        if (isLight) {
            document.documentElement.setAttribute('data-theme', 'light');
            document.body.classList.add('light-theme');
        } else {
            document.documentElement.removeAttribute('data-theme');
            document.body.classList.remove('light-theme');
        }

        if (themeToggleBtn) {
            const textEl = themeToggleBtn.querySelector('.theme-text');
            if (textEl) {
                textEl.textContent = isLight ? 'Dark Mode' : 'Light Mode';
            }
        }

        try {
            localStorage.setItem('churnguard_theme', theme);
        } catch(e) {}

        if (riskChartInstance || contractChartInstance) {
            renderAnalyticsCharts();
        }
    }

    // Initialize theme on load
    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const currentTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            applyTheme(newTheme);
            showToast(`Switched to ${newTheme === 'light' ? 'Light' : 'Dark'} Mode`, 'info');
        });
    }

    // -------------------------------------------------------------
    // 1. Tab Switching Logic
    // -------------------------------------------------------------
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = `tab-${btn.getAttribute('data-tab')}`;
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }

            if (btn.getAttribute('data-tab') === 'analytics-view') {
                renderAnalyticsCharts();
            }
            if (btn.getAttribute('data-tab') === 'history-view') {
                fetchPredictionHistory();
            }
            if (btn.getAttribute('data-tab') === 'registry-view') {
                fetchModelRegistry();
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Auto-compute Total Charges in Form
    // -------------------------------------------------------------
    const tenureInput = document.getElementById('tenure');
    const monthlyInput = document.getElementById('MonthlyCharges');
    const totalInput = document.getElementById('TotalCharges');

    function updateTotalCharges() {
        const tenure = parseFloat(tenureInput.value) || 0;
        const monthly = parseFloat(monthlyInput.value) || 0;
        if (!totalInput.value || totalInput.dataset.autoCalculated === 'true') {
            totalInput.value = (tenure * monthly).toFixed(2);
            totalInput.dataset.autoCalculated = 'true';
        }
    }

    tenureInput.addEventListener('input', updateTotalCharges);
    monthlyInput.addEventListener('input', updateTotalCharges);
    totalInput.addEventListener('input', () => {
        totalInput.dataset.autoCalculated = 'false';
    });

    // -------------------------------------------------------------
    // 3. Sample Data Presets
    // -------------------------------------------------------------
    const btnFillSample = document.getElementById('btnFillSample');
    const samples = [
        {
            customerID: 'CUST-HIGH-01',
            Contract: 'Month-to-month',
            PaymentMethod: 'Electronic check',
            tenure: 2,
            MonthlyCharges: 95.70,
            TotalCharges: 191.40,
            gender: 'Female',
            SeniorCitizen: 0,
            Partner: 'No',
            Dependents: 'No',
            InternetService: 'Fiber optic',
            TechSupport: 'No',
            OnlineSecurity: 'No',
            OnlineBackup: 'No',
            DeviceProtection: 'No',
            StreamingTV: 'Yes',
            PaperlessBilling: 'Yes'
        },
        {
            customerID: 'CUST-LOW-02',
            Contract: 'Two year',
            PaymentMethod: 'Credit card (automatic)',
            tenure: 64,
            MonthlyCharges: 45.20,
            TotalCharges: 2892.80,
            gender: 'Male',
            SeniorCitizen: 0,
            Partner: 'Yes',
            Dependents: 'Yes',
            InternetService: 'DSL',
            TechSupport: 'Yes',
            OnlineSecurity: 'Yes',
            OnlineBackup: 'Yes',
            DeviceProtection: 'Yes',
            StreamingTV: 'No',
            PaperlessBilling: 'No'
        }
    ];
    let sampleIdx = 0;

    btnFillSample.addEventListener('click', () => {
        const sample = samples[sampleIdx % samples.length];
        sampleIdx++;
        Object.keys(sample).forEach(key => {
            const el = document.getElementById(key);
            if (el) {
                el.value = sample[key];
            }
        });
        showToast('Sample customer profile loaded!', 'info');
    });

    // -------------------------------------------------------------
    // 4. Single Customer Prediction Submission
    // -------------------------------------------------------------
    const form = document.getElementById('singlePredictionForm');
    const btnPredict = document.getElementById('btnPredict');
    const predictSpinner = document.getElementById('predictSpinner');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultBody = document.getElementById('resultBody');
    const resultStatusBadge = document.getElementById('resultStatusBadge');
    const gaugeCircle = document.getElementById('gaugeCircle');
    const gaugePercentage = document.getElementById('gaugePercentage');
    const outcomeAlert = document.getElementById('outcomeAlert');
    const outcomeTitle = document.getElementById('outcomeTitle');
    const outcomeDesc = document.getElementById('outcomeDesc');
    const factorsList = document.getElementById('factorsList');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        btnPredict.disabled = true;
        predictSpinner.classList.remove('hidden');

        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => {
            if (key !== 'csrfmiddlewaretoken') {
                payload[key] = value;
            }
        });

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken') || ''
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || JSON.stringify(data.errors || 'Prediction failed'));
            }

            displaySinglePredictionResult(data);
            showToast(`Assessment complete: ${data.risk_level} Churn Risk`, data.badge_color === 'danger' ? 'error' : 'success');
            refreshGlobalStats();
        } catch (err) {
            showToast(`Error: ${err.message}`, 'error');
        } finally {
            btnPredict.disabled = false;
            predictSpinner.classList.add('hidden');
        }
    });

    function displaySinglePredictionResult(res) {
        resultPlaceholder.classList.add('hidden');
        resultBody.classList.remove('hidden');

        const pct = res.churn_percentage;
        gaugePercentage.textContent = `${pct}%`;

        let color = '#10b981'; // emerald
        let badgeClass = 'badge-success';
        let alertClass = 'success';

        if (res.risk_level === 'High') {
            color = '#f43f5e';
            badgeClass = 'badge-danger';
            alertClass = 'danger';
        } else if (res.risk_level === 'Medium') {
            color = '#f59e0b';
            badgeClass = 'badge-warning';
            alertClass = 'warning';
        }

        // Animated Gauge Gradient
        gaugeCircle.style.background = `conic-gradient(${color} 0% ${pct}%, rgba(255, 255, 255, 0.08) ${pct}% 100%)`;
        gaugePercentage.style.color = color;

        // Status Badge
        resultStatusBadge.className = `badge ${badgeClass}`;
        resultStatusBadge.textContent = `${res.risk_level} Risk Profile`;

        // Outcome Alert Box
        outcomeAlert.className = `outcome-alert ${alertClass}`;
        if (res.churn_predicted) {
            outcomeTitle.textContent = `High Risk: Churn Probability ${pct}%`;
            outcomeDesc.textContent = `This customer exhibits behavioural markers indicating impending cancellation. Immediate retention intervention recommended.`;
        } else {
            outcomeTitle.textContent = `Healthy Customer: Probability ${pct}%`;
            outcomeDesc.textContent = `Low churn probability detected. Subscriber engagement patterns indicate retention stability.`;
        }

        // Render Factors & Recommendations (Supports TreeSHAP & Legacy)
        factorsList.innerHTML = '';
        if (res.risk_factors && res.risk_factors.length > 0) {
            res.risk_factors.forEach(f => {
                const item = document.createElement('div');
                
                if (f.is_shap || typeof f.shap_value === 'number') {
                    // Modern TreeSHAP Directional Feature Attribution
                    const isIncrease = f.shap_value >= 0;
                    const containerClass = isIncrease ? 'risk-increase' : 'risk-decrease';
                    const sign = isIncrease ? '+' : '';
                    const tagClass = isIncrease ? 'positive' : 'negative';
                    const barWidth = Math.min(100, Math.max(15, Math.abs(f.shap_value) * 80));

                    item.className = `factor-item shap-item ${containerClass}`;
                    item.innerHTML = `
                        <div class="shap-meta-row">
                            <span>${f.factor}</span>
                            <span class="shap-value-tag ${tagClass}">${sign}${f.shap_value.toFixed(4)}</span>
                        </div>
                        <div class="shap-bar-track">
                            <div class="shap-bar-fill ${tagClass}" style="width: ${barWidth}%;"></div>
                        </div>
                        <div class="factor-detail">${f.detail}</div>
                    `;
                } else {
                    // Legacy Heuristic Format
                    const levelClass = f.impact.toLowerCase().includes('high') ? '' : (f.impact.toLowerCase().includes('medium') ? 'medium' : 'low');
                    item.className = `factor-item ${levelClass}`;
                    item.innerHTML = `
                        <div class="factor-header">
                            <span>${f.factor}</span>
                            <span>${f.impact}</span>
                        </div>
                        <div class="factor-detail">${f.detail}</div>
                    `;
                }
                factorsList.appendChild(item);
            });
        } else {
            factorsList.innerHTML = '<div class="factor-detail">No critical risk flags detected for this subscriber profile.</div>';
        }

        // Quick Meta
        document.getElementById('resCustomerId').textContent = res.customer_id || 'Anonymous';
        document.getElementById('resContract').textContent = res.contract || '-';
        document.getElementById('resMonthly').textContent = `$${parseFloat(res.monthly_charges).toFixed(2)}`;
        document.getElementById('resTenure').textContent = `${res.tenure} mo`;
    }

    // -------------------------------------------------------------
    // 5. Batch CSV Upload Handling & Celery Async Polling
    // -------------------------------------------------------------
    const dropzone = document.getElementById('dropzoneArea');
    const fileInput = document.getElementById('batchFileInput');
    const selectedFileName = document.getElementById('selectedFileName');
    const btnUploadBatch = document.getElementById('btnUploadBatch');
    const batchProgressWrapper = document.getElementById('batchProgressWrapper');
    const batchProgressBar = document.getElementById('batchProgressBar');
    const batchProgressPct = document.getElementById('batchProgressPct');
    const batchProgressLabel = document.getElementById('batchProgressLabel');
    const batchResultsSection = document.getElementById('batchResultsSection');
    const batchTableBody = document.getElementById('batchTableBody');

    let selectedFile = null;
    let batchPollInterval = null;

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        if (!file.name.endsWith('.csv') && !file.name.endsWith('.CSV')) {
            showToast('Please upload a valid .csv file.', 'error');
            return;
        }
        selectedFile = file;
        selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileName.classList.remove('hidden');
        btnUploadBatch.disabled = false;
    }

    btnUploadBatch.addEventListener('click', async () => {
        if (!selectedFile) return;

        btnUploadBatch.disabled = true;
        batchResultsSection.classList.add('hidden');
        batchProgressWrapper.classList.remove('hidden');
        batchProgressBar.style.width = '10%';
        batchProgressPct.textContent = '10%';
        batchProgressLabel.innerHTML = '<i class="fa-solid fa-gear fa-spin"></i> Submitting batch to Celery async queue...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/batch-predict/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken') || ''
                },
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Batch submission failed.');
            }

            if (data.status === 'COMPLETED') {
                // Was processed synchronously
                batchProgressBar.style.width = '100%';
                batchProgressPct.textContent = '100%';
                setTimeout(() => {
                    batchProgressWrapper.classList.add('hidden');
                    btnUploadBatch.disabled = false;
                }, 500);
                // Fetch full job data
                const statusRes = await fetch(`/api/batch-status/${data.job_id}/`);
                const fullJobData = await statusRes.json();
                renderBatchResults(fullJobData);
                showToast(`Batch processing complete! ${fullJobData.total_records} records evaluated.`, 'success');
                refreshGlobalStats();
            } else {
                // Async processing via Celery — poll status endpoint
                pollBatchJob(data.job_id);
            }
        } catch (err) {
            batchProgressWrapper.classList.add('hidden');
            btnUploadBatch.disabled = false;
            showToast(`Batch error: ${err.message}`, 'error');
        }
    });

    function pollBatchJob(jobId) {
        if (batchPollInterval) clearInterval(batchPollInterval);

        batchPollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/batch-status/${jobId}/`);
                const job = await res.json();

                const pct = job.progress_percentage || 0;
                batchProgressBar.style.width = `${pct}%`;
                batchProgressPct.textContent = `${pct}%`;

                if (job.status === 'PROCESSING') {
                    batchProgressLabel.innerHTML = `<i class="fa-solid fa-brain fa-spin"></i> Scoring batch with ML pipeline (${pct}%)...`;
                } else if (job.status === 'COMPLETED') {
                    clearInterval(batchPollInterval);
                    batchProgressBar.style.width = '100%';
                    batchProgressPct.textContent = '100%';
                    batchProgressLabel.innerHTML = `<i class="fa-solid fa-circle-check text-emerald"></i> Batch complete!`;
                    
                    setTimeout(() => {
                        batchProgressWrapper.classList.add('hidden');
                        btnUploadBatch.disabled = false;
                    }, 600);

                    renderBatchResults(job);
                    showToast(`Batch finished! ${job.total_records} records scored.`, 'success');
                    refreshGlobalStats();
                } else if (job.status === 'FAILED') {
                    clearInterval(batchPollInterval);
                    batchProgressWrapper.classList.add('hidden');
                    btnUploadBatch.disabled = false;
                    showToast(`Batch failed: ${job.error_message || 'Unknown error'}`, 'error');
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        }, 1200);
    }

    function renderBatchResults(data) {
        batchResultsSection.classList.remove('hidden');
        document.getElementById('batchTotalCount').textContent = data.total_records;
        document.getElementById('batchChurnCount').textContent = data.churn_detected;
        document.getElementById('batchChurnRate').textContent = `${data.churn_rate}%`;
        document.getElementById('batchHighCount').textContent = data.high_risk_count;
        document.getElementById('batchLowCount').textContent = data.low_risk_count;

        batchTableBody.innerHTML = '';
        data.preview.forEach(row => {
            const tr = document.createElement('tr');
            const riskBadge = row.risk_level === 'High' ? 'badge-danger' : (row.risk_level === 'Medium' ? 'badge-warning' : 'badge-success');
            const actionStatus = row.churn_predicted
                ? '<span class="status-indicator churned"><i class="fa-solid fa-circle-xmark"></i> Churn Risk</span>'
                : '<span class="status-indicator retained"><i class="fa-solid fa-circle-check"></i> Retained</span>';

            tr.innerHTML = `
                <td><strong>${row.customer_id}</strong></td>
                <td>${row.contract_type}</td>
                <td>${row.tenure} mo</td>
                <td>$${row.monthly_charges.toFixed(2)}</td>
                <td>${(row.churn_probability * 100).toFixed(1)}%</td>
                <td><span class="badge ${riskBadge}">${row.risk_level}</span></td>
                <td>${actionStatus}</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    // -------------------------------------------------------------
    // 6. Visual Intelligence Charts (Chart.js)
    // -------------------------------------------------------------
    async function renderAnalyticsCharts() {
        try {
            const chartCanvasRisk = document.getElementById('riskDistributionChart');
            const chartCanvasContract = document.getElementById('contractChart');
            if (!chartCanvasRisk || !chartCanvasContract) return;

            const res = await fetch('/api/stats/');
            const data = await res.json();

            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            const textColor = isLight ? '#475569' : '#94a3b8';
            const gridColor = isLight ? 'rgba(0, 0, 0, 0.07)' : 'rgba(255, 255, 255, 0.05)';
            const donutBorder = isLight ? '#ffffff' : '#0f172a';

            // Doughnut Chart: Risk Distribution
            const ctxRisk = chartCanvasRisk.getContext('2d');
            if (riskChartInstance) riskChartInstance.destroy();

            riskChartInstance = new Chart(ctxRisk, {
                type: 'doughnut',
                data: {
                    labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                    datasets: [{
                        data: [data.high_risk, data.medium_risk, data.low_risk],
                        backgroundColor: ['#f43f5e', '#f59e0b', '#10b981'],
                        borderColor: donutBorder,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 12 } }
                        }
                    }
                }
            });

            // Bar Chart: Contract Breakdown
            const ctxContract = chartCanvasContract.getContext('2d');
            if (contractChartInstance) contractChartInstance.destroy();

            const contractLabels = data.contracts ? data.contracts.map(c => c.contract_type) : ['Month-to-month', 'One year', 'Two year'];
            const contractTotal = data.contracts ? data.contracts.map(c => c.count) : [0, 0, 0];
            const contractChurn = data.contracts ? data.contracts.map(c => c.churn_count) : [0, 0, 0];

            contractChartInstance = new Chart(ctxContract, {
                type: 'bar',
                data: {
                    labels: contractLabels,
                    datasets: [
                        {
                            label: 'Total Customers',
                            data: contractTotal,
                            backgroundColor: isLight ? 'rgba(79, 70, 229, 0.7)' : 'rgba(99, 102, 241, 0.6)',
                            borderColor: isLight ? '#4f46e5' : '#6366f1',
                            borderWidth: 1
                        },
                        {
                            label: 'Predicted Churn',
                            data: contractChurn,
                            backgroundColor: isLight ? 'rgba(225, 29, 72, 0.85)' : 'rgba(244, 63, 94, 0.8)',
                            borderColor: isLight ? '#e11d48' : '#f43f5e',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: textColor }, grid: { color: gridColor } },
                        y: { ticks: { color: textColor }, grid: { color: gridColor } }
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 12 } }
                        }
                    }
                }
            });
        } catch (e) {
            console.error('Failed to load chart analytics:', e);
        }
    }

    // -------------------------------------------------------------
    // 7. Prediction History Log & Filters
    // -------------------------------------------------------------
    const historySearch = document.getElementById('historySearchInput');
    const historyFilter = document.getElementById('historyRiskFilter');
    const btnReloadHistory = document.getElementById('btnReloadHistory');
    const historyTableBody = document.getElementById('historyTableBody');

    async function fetchPredictionHistory() {
        const risk = historyFilter.value;
        const search = historySearch.value.trim();
        let url = `/api/history/?risk=${encodeURIComponent(risk)}&limit=100`;
        if (search) url += `&search=${encodeURIComponent(search)}`;

        try {
            const res = await fetch(url);
            const data = await res.json();

            historyTableBody.innerHTML = '';
            if (data.length === 0) {
                historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-cell">No matching prediction records found.</td></tr>`;
                return;
            }

            data.forEach(r => {
                const tr = document.createElement('tr');
                const riskBadge = r.risk_level === 'High' ? 'badge-danger' : (r.risk_level === 'Medium' ? 'badge-warning' : 'badge-success');
                const status = r.churn_predicted
                    ? '<span class="status-indicator churned"><i class="fa-solid fa-circle-xmark"></i> Will Churn</span>'
                    : '<span class="status-indicator retained"><i class="fa-solid fa-circle-check"></i> Retained</span>';

                tr.innerHTML = `
                    <td>${r.created_at_formatted}</td>
                    <td><strong>${r.customer_id}</strong></td>
                    <td>${r.contract_type}</td>
                    <td>${r.tenure} mo</td>
                    <td>$${parseFloat(r.monthly_charges).toFixed(2)}</td>
                    <td>${parseFloat(r.churn_probability).toFixed(2)}</td>
                    <td><span class="badge ${riskBadge}">${r.risk_level}</span></td>
                    <td>${status}</td>
                `;
                historyTableBody.appendChild(tr);
            });
        } catch (e) {
            console.error('Error fetching history:', e);
        }
    }

    historySearch.addEventListener('input', debounce(fetchPredictionHistory, 300));
    historyFilter.addEventListener('change', fetchPredictionHistory);
    btnReloadHistory.addEventListener('click', () => {
        fetchPredictionHistory();
        showToast('Audit log updated.', 'info');
    });

    // -------------------------------------------------------------
    // 8. Model Registry & Benchmark Controller
    // -------------------------------------------------------------
    const registryTableBody = document.getElementById('registryTableBody');
    const btnRefreshRegistry = document.getElementById('btnRefreshRegistry');

    async function fetchModelRegistry() {
        if (!registryTableBody) return;
        try {
            const res = await fetch('/api/models/');
            const data = await res.json();

            registryTableBody.innerHTML = '';
            if (data.length === 0) {
                registryTableBody.innerHTML = `<tr><td colspan="9" class="text-center empty-cell">No registered models found. Run ml_engine/train.py to train and register models.</td></tr>`;
                return;
            }

            data.forEach(m => {
                const tr = document.createElement('tr');
                const actionCell = m.is_active
                    ? `<span class="badge-active-model"><i class="fa-solid fa-circle-check"></i> Production Active</span>`
                    : `<button class="btn-activate" data-id="${m.id}" data-name="${m.name}" data-auc="${m.roc_auc}"><i class="fa-solid fa-bolt"></i> Deploy Model</button>`;

                tr.innerHTML = `
                    <td><strong>${m.name}</strong></td>
                    <td><span class="badge">${m.algorithm}</span></td>
                    <td>${m.imbalance_strategy}</td>
                    <td><strong>${m.roc_auc.toFixed(4)}</strong></td>
                    <td>${m.pr_auc.toFixed(4)}</td>
                    <td>${m.f1_churn.toFixed(4)}</td>
                    <td>${(m.recall_churn * 100).toFixed(1)}%</td>
                    <td>${m.trained_at_formatted}</td>
                    <td>${actionCell}</td>
                `;
                registryTableBody.appendChild(tr);
            });

            // Attach activation click handlers
            registryTableBody.querySelectorAll('.btn-activate').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const id = btn.getAttribute('data-id');
                    const modelName = btn.getAttribute('data-name');
                    const modelAuc = btn.getAttribute('data-auc');
                    btn.disabled = true;
                    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Activating...`;

                    try {
                        const actRes = await fetch(`/api/models/${id}/activate/`, {
                            method: 'POST',
                            headers: {
                                'X-CSRFToken': getCookie('csrftoken') || ''
                            }
                        });
                        const actData = await actRes.json();
                        if (actRes.ok) {
                            showToast(`Active model switched to ${modelName}!`, 'success');
                            // Update navbar display
                            const navName = document.getElementById('navActiveModelName');
                            const navMetric = document.getElementById('navActiveModelMetric');
                            if (navName) navName.textContent = modelName;
                            if (navMetric) navMetric.textContent = `ROC-AUC: ${modelAuc}`;
                            fetchModelRegistry();
                        } else {
                            showToast(`Activation failed: ${actData.error || 'Unknown error'}`, 'error');
                        }
                    } catch (err) {
                        showToast(`Activation error: ${err.message}`, 'error');
                    }
                });
            });
        } catch (e) {
            console.error('Error loading model registry:', e);
        }
    }

    if (btnRefreshRegistry) {
        btnRefreshRegistry.addEventListener('click', () => {
            fetchModelRegistry();
            showToast('Model registry refreshed.', 'info');
        });
    }

    // -------------------------------------------------------------
    // 9. Live Global Stats Refresh
    // -------------------------------------------------------------
    const refreshStatsBtn = document.getElementById('refreshStatsBtn');
    if (refreshStatsBtn) {
        refreshStatsBtn.addEventListener('click', () => {
            refreshGlobalStats();
            showToast('Stats synced.', 'info');
        });
    }

    async function refreshGlobalStats() {
        try {
            const res = await fetch('/api/stats/');
            const data = await res.json();
            document.getElementById('kpiTotal').textContent = data.total;
            document.getElementById('kpiHighRisk').textContent = data.high_risk;
            document.getElementById('kpiChurnRate').textContent = data.churn_rate;
            document.getElementById('kpiAvgTenure').innerHTML = `${data.avg_tenure} <span class="unit">mo</span>`;
            document.getElementById('kpiAvgCharges').textContent = `$${data.avg_charges}`;
        } catch (e) {
            console.error('Failed to refresh stats:', e);
        }
    }

    // -------------------------------------------------------------
    // Helper Utilities
    // -------------------------------------------------------------
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const icon = type === 'success' ? 'circle-check' : (type === 'error' ? 'circle-exclamation' : 'circle-info');
        toast.innerHTML = `<i class="fa-solid fa-${icon}"></i> <span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    function debounce(fn, delay) {
        let timer = null;
        return function(...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    }
});
