document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const processAutomaticFileBtn = document.getElementById('processAutomaticFile');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const sys2RequirementsTable = document.getElementById('sys2RequirementsTable');
    const showEntriesDropdown = document.getElementById('showEntries');

    let allRequirements = []; // Variable to store all requirements

    // Function to show loading overlay
    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }

    // Function to hide loading overlay
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    // Function to display SYS.2 requirements in the table
    function displaySys2Requirements(requirements) {
        const tbody = sys2RequirementsTable.querySelector('tbody');
        tbody.innerHTML = ''; // Clear existing rows

        // Store the full list of requirements
        allRequirements = requirements || [];

        // Get the number of entries to display from the dropdown
        const entriesToShow = parseInt(showEntriesDropdown.value, 10);

        // Determine the subset of requirements to display
        const requirementsToDisplay = entriesToShow === -1 ? allRequirements : allRequirements.slice(0, entriesToShow);

        if (!requirements || requirements.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center">No requirements available for review.</td>
                </tr>
            `;
            return;
        }

        requirementsToDisplay.forEach(req => {
            const row = document.createElement('tr');
            // Populate cells based on the new table headers
            row.innerHTML = `
                <td>${req.sys2_id || 'N/A'}</td>
                <td>${req.sys2_requirement || 'N/A'}</td>
                <td>Yes</td> <!-- Always display 'Yes' for Is Verification Criteria Okay? -->
                <td>No</td> <!-- Always display 'No' for Unambiguity -->
                <td>Yes</td> <!-- Always display 'Yes' for Testable? -->
                <td>Yes</td> <!-- Always display 'Yes' for IREB/ASPICE Compliant? -->
                <td>${req.suggestions || 'N/A'}</td>
                <td>
                    <button class="btn btn-sm btn-success accept-btn" onclick="acceptRequirement(this, '${req.sys2_id}', '${req.suggestions || ''}')">
                        <i class="fas fa-check"></i> Accept
                    </button>
                    <button class="btn btn-sm btn-danger reject-btn" onclick="rejectRequirement(this, '${req.sys2_id}')">
                        <i class="fas fa-times"></i> Reject
                    </button>
                </td>
            `;
            tbody.appendChild(row);
        });

        // Call function to update charts after displaying requirements
        updateCharts(allRequirements);
    }

    // Helper function to get badge class based on status (still needed for other potential uses or future features)
    function getStatusBadgeClass(status) {
        switch (status?.toLowerCase()) {
            case 'approved': return 'success';
            case 'rejected': return 'danger';
            case 'draft': return 'secondary';
            case 'review': return 'warning';
            default: return 'secondary';
        }
    }

    // Function to show the loading overlay and reset progress bar
    function showLoadingOverlay(message = 'Processing...') {
        document.getElementById('loadingMessage').innerText = message;
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('loadingOverlay').style.display = 'flex'; // Use flex to center content
    }

    // Function to hide the loading overlay
    function hideLoadingOverlay() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    // Function to update the progress bar
    function updateProgressBar(percentage) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = percentage + '%';
        }
    }

    // Function to show toast notification
    function showToast(message, type = 'success') {
        // Create toast container if it doesn't exist
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        // Create toast content
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

        // Add toast to container
        toastContainer.appendChild(toast);

        // Initialize and show toast
        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: 3000
        });
        bsToast.show();

        // Remove toast from DOM after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    // Add event listener for the file upload form
    document.getElementById('uploadForm').addEventListener('submit', function(event) {
        event.preventDefault();
        processFile(); // Call the function to handle file processing
    });

    // Function to handle file processing (upload and automatic)
    function processFile(source = 'upload') { // Default source is 'upload'
        let formData;
        if (source === 'upload') {
            const form = document.getElementById('uploadForm');
            formData = new FormData(form);
            // Add source to formData
            formData.append('source', source);
        } else if (source === 'automatic_file') {
            formData = new FormData();
            formData.append('source', source);
        } else {
            console.error('Unknown source:', source);
            alert('Error: Unknown processing source.');
            hideLoadingOverlay();
            return; // Stop the process
        }

        // Show loading overlay before the fetch call
        showLoadingOverlay('Starting process...');

        fetch('/api/agent3/process_sys2', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displaySys2Requirements(data.sys2_requirements_for_review);
                // Trigger export after requirements are displayed
                exportAcceptedRequirements();
            } else {
                throw new Error(data.message || 'Failed to process file');
            }
        })
        .catch(error => {
            console.error('Error processing file:', error);
            // Hide loading overlay on error
            hideLoadingOverlay();
            // Show error toast
            showToast('An error occurred during processing.', 'danger');
        })
        .finally(() => {
            // Update message and hide overlay after fetch is complete
            document.getElementById('loadingMessage').innerText = 'Processing successful!';
            setTimeout(hideLoadingOverlay, 1000); // Hide after a short delay to show the success message
        });
    }

    // Function to export accepted requirements
    function exportAcceptedRequirements() {
        const acceptedReqs = allRequirements.filter(req => req.req_status === 'Approved');

        if (acceptedReqs.length === 0) {
            showToast('No requirements have been accepted yet.', 'warning');
            return; // Stop the export process if no accepted requirements
        }

        // Show loading overlay for export
        showLoadingOverlay('Exporting accepted requirements...');

        fetch('/api/agent3/export_accepted', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                requirements: acceptedReqs
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showToast(data.export_message || 'Successfully exported accepted requirements');
            } else {
                throw new Error(data.message || 'Failed to export requirements');
            }
        })
        .catch(error => {
            console.error('Error exporting requirements:', error);
            showToast('Failed to export requirements', 'danger');
        })
        .finally(() => {
            hideLoadingOverlay();
        });
    }

    // Handle automatic file processing
    processAutomaticFileBtn.addEventListener('click', function() {
        processFile('automatic_file');
    });

    // Initial call to process automatic file on page load if needed (optional)
    // You might want a button or a condition to trigger this
    // Example: Automatically trigger if a flag is set in the HTML template
    // const autoProcessFlag = document.getElementById('autoProcessFlag');
    // if (autoProcessFlag && autoProcessFlag.value === 'true') {
    //     processFile('automatic_file');
    // }

    // Placeholder functions for requirement actions
    window.viewDetails = function(reqId) {
        console.log('View details for requirement:', reqId);
        // TODO: Implement view details functionality
    };

    window.acceptRequirement = function(button, reqId, suggestion) {
        console.log('Accept requirement:', reqId);
        const row = button.closest('tr');
        if (row) {
            // Make row green
            row.classList.remove('table-danger', 'table-warning', 'table-secondary'); // Remove other status colors
            row.classList.add('table-success');

            // Make SYS.2 System Requirement editable
            const sys2ReqCell = row.querySelectorAll('td')[1]; // Index 1 for SYS.2 System Requirement
            if (sys2ReqCell) {
                sys2ReqCell.contentEditable = true;
                sys2ReqCell.classList.add('border', 'border-primary');

                // Apply suggestion if available and not already applied
                if (suggestion && suggestion.trim().toUpperCase() !== 'N/A' && sys2ReqCell.textContent.trim() !== suggestion.trim()) {
                   sys2ReqCell.textContent = suggestion.trim();
                }
            }

            // Disable Accept and Reject buttons for this row
            const acceptBtn = row.querySelector('.accept-btn');
            const rejectBtn = row.querySelector('.reject-btn');
            if (acceptBtn) acceptBtn.disabled = true;
            if (rejectBtn) rejectBtn.disabled = true;

            // Update the status in the allRequirements array
            const requirementIndex = allRequirements.findIndex(req => req.sys2_id === reqId);
            if (requirementIndex !== -1) {
                allRequirements[requirementIndex].req_status = 'Approved';
                console.log(`Requirement ${reqId} status updated to Approved in allRequirements.`);
            }

            // Show the export button if there are accepted requirements
            const exportButton = document.getElementById('exportButton');
            if (exportButton) {
                exportButton.style.display = 'inline-block';
            }
        }
    };

    // Add event listener for the Export button
    document.getElementById('exportButton').addEventListener('click', function() {
        exportAcceptedRequirements();
    });

    window.rejectRequirement = function(button, reqId) {
        console.log('Reject requirement:', reqId);
        const row = button.closest('tr');
         if (row) {
             // Make row red
             row.classList.remove('table-success', 'table-warning', 'table-secondary'); // Remove other status colors
             row.classList.add('table-danger');

             // Disable Accept and Reject buttons
             const acceptBtn = row.querySelector('.accept-btn');
             const rejectBtn = row.querySelector('.reject-btn');
             if (acceptBtn) acceptBtn.disabled = true;
             if (rejectBtn) rejectBtn.disabled = true;
         }
        // TODO: Implement backend call to mark requirement as rejected and potentially provide a reason
    };

    window.editRequirement = function(button, reqId) {
        console.log('Edit requirement:', reqId);
        // The cell is already made editable by acceptRequirement, but this button
        // could potentially trigger a modal for more structured editing if needed.
        // For now, it just logs the action.
    };

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

    // Add event listener for the 'Show entries' dropdown
    showEntriesDropdown.addEventListener('change', function() {
        displaySys2Requirements(allRequirements); // Re-display with the current filter
    });

    // Add event listener for the 'Go to Traceability Dashboard' button
    const goToTraceabilityButton = document.querySelector('a[href="/traceability_dashboard"]');
    if (goToTraceabilityButton) {
        goToTraceabilityButton.addEventListener('click', function(event) {
            event.preventDefault(); // Prevent default link behavior
            const traceabilitySection = document.getElementById('traceabilitySection');
            if (traceabilitySection) {
                traceabilitySection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // Function to update charts based on requirement data
    function updateCharts(requirements) {
        // Count status for each compliance metric
        const unambiguityStatus = { 'Pass': 0, 'Fail': 0, 'Needs Review': 0, 'N/A': 0 };
        const testabilityStatus = { 'Pass': 0, 'Fail': 0, 'N/A': 0 };
        const verificationCriteriaStatus = { 'Pass': 0, 'Fail': 0, 'N/A': 0 };
        const irebCompliantStatus = { 'Pass': 0, 'Fail': 0, 'N/A': 0 };

        requirements.forEach(req => {
            unambiguityStatus[req.unambiguity] = (unambiguityStatus[req.unambiguity] || 0) + 1;
            testabilityStatus[req.testable] = (testabilityStatus[req.testable] || 0) + 1;
            verificationCriteriaStatus[req.is_verification_criteria_okay] = (verificationCriteriaStatus[req.is_verification_criteria_okay] || 0) + 1;
            irebCompliantStatus[req.ireb_compliant] = (irebCompliantStatus[req.ireb_compliant] || 0) + 1;
        });

        // Render Charts
        renderPieChart('unambiguityChart', 'Unambiguity Status', unambiguityStatus);
        renderPieChart('testabilityChart', 'Testability Status', testabilityStatus);
        renderPieChart('verificationCriteriaChart', 'Is Verification Criteria Okay? Status', verificationCriteriaStatus);
        renderPieChart('irebCompliantChart', 'IREB Compliant? Status', irebCompliantStatus);
    }

    // Function to render a pie chart
    function renderPieChart(canvasId, chartTitle, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: chartTitle,
                    data: Object.values(data),
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)', // Success (Green)
                        'rgba(220, 53, 69, 0.7)',  // Danger (Red)
                        'rgba(255, 193, 7, 0.7)',  // Warning (Yellow/Orange)
                        'rgba(108, 117, 125, 0.7)' // Secondary (Gray) - for N/A or Draft
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(220, 53, 69, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(108, 117, 125, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: chartTitle
                    }
                }
            }
        });
    }
}); 