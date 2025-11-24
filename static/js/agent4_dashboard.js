document.addEventListener('DOMContentLoaded', function() {
    console.log('Agent 4 dashboard JavaScript loaded');
    
    const exportButton = document.getElementById('exportButton');
    console.log('Export button found:', exportButton);

    if (exportButton) {
        exportButton.addEventListener('click', function() {
            console.log('Export button clicked');
            
            // Show loading state
            exportButton.disabled = true;
            exportButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';

            fetch('/api/agent4/export_test_cases', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                console.log('Export response:', response);
                return response.json();
            })
            .then(data => {
                console.log('Export data:', data);
                if (data.status === 'success') {
                    alert('Test cases exported successfully to D:\\AgentX\\WHALE_AI_AGENTS\\Inputs\\sys.5_test_cases.xlsx');
                } else {
                    alert('Error: ' + (data.message || 'Failed to export test cases'));
                }
            })
            .catch(error => {
                console.error('Export error:', error);
                alert('An error occurred while exporting test cases. Please check the console for details.');
            })
            .finally(() => {
                // Reset button state
                exportButton.disabled = false;
                exportButton.innerHTML = '<i class="fas fa-file-export"></i> Export Test Cases';
            });
        });
    } else {
        console.error('Export button not found in the DOM');
    }

    // Dark Mode Toggle Functionality
    const darkModeToggle = document.getElementById('darkModeToggle');
    const currentTheme = localStorage.getItem('theme');

    if (currentTheme) {
        document.body.classList.add(currentTheme);
        if (currentTheme === 'dark-mode') {
            darkModeToggle.checked = true;
        }
    }

    darkModeToggle.addEventListener('change', () => {
        if (darkModeToggle.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light-mode');
        }
    });

    // Handle Process Automatic File button
    const processBtn = document.getElementById('processAutomaticFile');
    const statusDiv = document.getElementById('processingStatusAgent4');
    const progressBar = document.getElementById('processingProgressBarAgent4');

    if (processBtn) {
        processBtn.addEventListener('click', function() {
            // Show progress bar and status
            statusDiv.classList.remove('d-none');
            progressBar.classList.remove('d-none');

            processBtn.disabled = true;
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            fetch('/api/agent4/load_sys2')
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') {
                        renderTestCases(data.test_cases);
                        alert('Test cases generated successfully!');
                    } else {
                        alert(data.message || 'Failed to load file.');
                    }
                })
                .catch(error => {
                    alert('Error processing file: ' + error);
                })
                .finally(() => {
                    // Hide progress bar and status
                    statusDiv.classList.add('d-none');
                    progressBar.classList.add('d-none');
                    processBtn.disabled = false;
                    processBtn.innerHTML = 'Process Automatic File';
                });
        });
    }

    // Render traceability charts and table
    function renderTraceability(testCases) {
        // Pie chart for SYS.2 Req. ID Coverage
        const sys2Counts = {};
        testCases.forEach(tc => {
            const sys2id = tc['SYS.2 Req. ID'] || 'Unknown';
            sys2Counts[sys2id] = (sys2Counts[sys2id] || 0) + 1;
        });
        const sys2Labels = Object.keys(sys2Counts);
        const sys2Data = Object.values(sys2Counts);

        // Pie chart for Test Case ID Distribution
        const testCaseLabels = testCases.map(tc => tc['Test Case ID'] || '');
        const testCaseData = testCaseLabels.map(() => 1);

        // Render charts (assumes Chart.js is loaded and canvas elements exist)
        if (window.sys2PieChart && typeof window.sys2PieChart.destroy === 'function') window.sys2PieChart.destroy();
        if (window.testCasePieChart && typeof window.testCasePieChart.destroy === 'function') window.testCasePieChart.destroy();

        const sys2Ctx = document.getElementById('sys2PieChart').getContext('2d');
        const testCaseCtx = document.getElementById('testCasePieChart').getContext('2d');

        window.sys2PieChart = new Chart(sys2Ctx, {
            type: 'pie',
            data: {
                labels: sys2Labels,
                datasets: [{
                    data: sys2Data,
                    backgroundColor: sys2Labels.map((_, i) => `hsl(${i * 360 / sys2Labels.length}, 70%, 60%)`),
                }]
            }
        });

        window.testCasePieChart = new Chart(testCaseCtx, {
            type: 'pie',
            data: {
                labels: testCaseLabels,
                datasets: [{
                    data: testCaseData,
                    backgroundColor: testCaseLabels.map((_, i) => `hsl(${i * 360 / testCaseLabels.length}, 70%, 60%)`),
                }]
            }
        });

        // Populate traceability table
        const tbody = document.querySelector('#traceabilityTable tbody');
        tbody.innerHTML = '';
        testCases.forEach(tc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${tc['SYS.2 Req. ID'] || ''}</td>
                <td>${tc['Test Case ID'] || ''}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // Update renderTestCases to also update traceability
    function renderTestCases(testCases) {
        const tableBody = document.querySelector('table tbody');
        tableBody.innerHTML = '';
        if (!testCases || testCases.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="9" class="text-center">No test cases generated yet.</td></tr>';
            renderTraceability([]);
            return;
        }
        testCases.forEach(tc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${tc['SYS.2 Req. ID'] || ''}</td>
                <td>${tc['SYS.2 System Requirement'] || ''}</td>
                <td>${tc['Test Case ID'] || ''}</td>
                <td>${tc['Description'] || ''}</td>
                <td>${tc['Preconditions'] || ''}</td>
                <td>${(tc['Test Steps'] || '').replace(/\n/g, '<br>')}</td>
                <td>${(tc['Expected Results'] || '').replace(/\n/g, '<br>')}</td>
                <td>${tc['Pass/Fail Criteria'] || ''}</td>
                <td>${tc['Priority'] || ''}</td>
            `;
            tableBody.appendChild(row);
        });
        renderTraceability(testCases);
    }

    // Optionally, load test cases on page load
    fetch('/api/agent4/test_cases')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                renderTestCases(data.test_cases);
            }
        });

    // Smooth scroll to traceability dashboard
    const goToTraceabilityBtn = document.getElementById('goToTraceabilityBtn');
    if (goToTraceabilityBtn) {
        goToTraceabilityBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const traceabilitySection = document.getElementById('traceabilityDashboard');
            if (traceabilitySection) {
                traceabilitySection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // TODO: Add other Agent 4 specific JavaScript functionalities here
}); 