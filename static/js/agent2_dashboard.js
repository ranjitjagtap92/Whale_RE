// JavaScript for Agent 2 Dashboard

document.addEventListener('DOMContentLoaded', () => {
    const processAutomaticFileBtn = document.getElementById('processAutomaticFileBtn');
    const showManualInputBtn = document.getElementById('showManualInputBtn');
    const manualInputCard = document.getElementById('manualInputCard');
    const processManualInputBtn = document.getElementById('processManualInputBtn');
    const fileInput = document.getElementById('fileInputAgent2');
    const rawContentInput = document.getElementById('rawContentAgent2');
    const processingStatusDiv = document.getElementById('processingStatusAgent2');
    const manualProcessingStatusDiv = document.getElementById('manualProcessingStatusAgent2');
    const sys2RequirementsTableContainer = document.getElementById('sys2RequirementsTableContainer');
    const sys2RequirementsPlaceholder = document.getElementById('sys2RequirementsPlaceholder');
    const progressBar = document.getElementById('processingProgressBarAgent2');

    // Event listener for processing the automatic file
    processAutomaticFileBtn.addEventListener('click', () => {
        // Hide manual input section if visible
        manualInputCard.style.display = 'none';
        processInput('automatic_file');
    });

    // Event listener to show manual input section
    showManualInputBtn.addEventListener('click', () => {
        manualInputCard.style.display = 'block';
    });

    // Event listener for processing manual input
    processManualInputBtn.addEventListener('click', () => {
        processInput('manual_input');
    });


    // Function to handle input processing (either automatic file or manual input)
    function processInput(source) {
        let formData = new FormData();
        let hasInput = false;
        const statusDiv = source === 'automatic_file' ? processingStatusDiv : manualProcessingStatusDiv;

        if (source === 'automatic_file') {
            formData.append('source', 'automatic_file');
            hasInput = true;
            statusDiv.textContent = 'Processing automatic SYS.1 file...';
            statusDiv.classList.remove('d-none', 'alert-success', 'alert-danger');
            statusDiv.classList.add('alert-info');
             // manualProcessingStatusDiv.classList.add('d-none'); // Hide manual status if showing
            sendToAgent2Process(formData, statusDiv);

        } else if (source === 'manual_input') {
            const files = fileInput.files;
            const rawContent = rawContentInput.value;

            if (files.length === 0 && rawContent.trim() === '') {
                alert('Please select a file or enter raw content.');
                return;
            }

            for (const file of files) {
                formData.append('file', file);
            }
            formData.append('raw_content', rawContent);
            hasInput = true;

            statusDiv.textContent = 'Uploading and processing manual input...';
            statusDiv.classList.remove('d-none', 'alert-success', 'alert-danger');
            statusDiv.classList.add('alert-info');
             // processingStatusDiv.classList.add('d-none'); // Hide automatic status if showing
            progressBar.classList.remove('d-none'); // Show progress bar

            // --- Send manual input to /api/upload first (like Agent 1) ---
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    statusDiv.textContent = 'Upload successful. Processing SYS.1 requirements...';
                    // --- Now trigger Agent 2's main processing which reads from session ---
                    // Agent 2's /api/agent2/process should be updated to primarily read from session if possible,
                    // or we can pass a flag. For now, let's just call it without formData
                    // and ensure the backend reads from session if no file/raw_content is provided.
                    // A better approach is to explicitly tell the backend to process session data.
                    
                    // Let's assume the backend /api/agent2/process, when called without file/raw_content
                    // and with source='session', will read from session.
                    const agent2ProcessFormData = new FormData();
                    agent2ProcessFormData.append('source', 'session'); // Indicate source is session
                    sendToAgent2Process(agent2ProcessFormData, statusDiv); // Call Agent 2 process with session source

                } else {
                    statusDiv.classList.remove('alert-info', 'alert-success');
                    statusDiv.classList.add('alert-danger');
                    statusDiv.textContent = 'Upload Error: ' + (data.message || 'Unknown error');
                    if (sys2RequirementsPlaceholder) sys2RequirementsPlaceholder.style.display = 'block';
                }
            })
            .catch(error => {
                statusDiv.classList.remove('alert-info', 'alert-success');
                statusDiv.classList.add('alert-danger');
                statusDiv.textContent = 'Upload Fetch Error: ' + error.message;
                 if (sys2RequirementsPlaceholder) sys2RequirementsPlaceholder.style.display = 'block';
            });

        }

        if (!hasInput) return;

        // Hide initial placeholder
        if (sys2RequirementsPlaceholder) sys2RequirementsPlaceholder.style.display = 'none';
        // Clear previous results
        const table = document.getElementById('sys2RequirementsTable');
        const tbody = table.querySelector('tbody');
        tbody.innerHTML = '';
    }

    // Function to send data to /api/agent2/process
    function sendToAgent2Process(formData, statusDiv) {
        progressBar.classList.remove('d-none'); // Show progress bar

        fetch('/api/agent2/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            progressBar.classList.add('d-none'); // Hide progress bar on completion
            if (data.status === 'success') {
                statusDiv.textContent = data.message || 'Processing successful.';

                // Display SYS.2 requirements and other data
                displaySys2Requirements(data.sys2_requirements);
                // displayDependencies(data.dependencies); // Implement this function
                // displayClassification(data.classification); // Implement this function
                // displayVerificationMapping(data.verification_mapping); // Implement this function

                // --- Call function to update dashboard charts and summary ---
                fetchAndDisplayCombinedDashboardSummary();
                // -------------------------------------------------------------

                // --- Show the traceability dashboard section ---
                const traceabilitySection = document.getElementById('sys2TraceabilityDashboardSection');
                if (traceabilitySection) {
                    traceabilitySection.classList.remove('hidden-dashboard-section');
                }
                // -----------------------------------------------

                // --- Save SYS.2 requirements XLSX on the server and show notification ---
                fetch('/api/agent2/save_sys2_xlsx', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        const toastEl = document.getElementById('exportToast');
                        if (toastEl) {
                            const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
                            toastEl.querySelector('.toast-body').textContent = data.status === 'success'
                                ? 'SYS.2 requirements saved to server!'
                                : 'Failed to save SYS.2 requirements: ' + data.message;
                            toast.show();
                        }
                    })
                    .catch(error => {
                        const toastEl = document.getElementById('exportToast');
                        if (toastEl) {
                            const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
                            toastEl.querySelector('.toast-body').textContent = 'Error saving SYS.2 requirements: ' + error.message;
                            toast.show();
                        }
                    });
                // -------------------------------------------------------------------

            } else {
                statusDiv.classList.remove('alert-info', 'alert-success');
                statusDiv.classList.add('alert-danger');
                statusDiv.textContent = 'Error: ' + (data.message || 'Unknown error');
                // Show placeholder if there's an error
                 if (sys2RequirementsPlaceholder) sys2RequirementsPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            progressBar.classList.add('d-none'); // Hide progress bar on error
            statusDiv.classList.remove('alert-info', 'alert-success');
            statusDiv.classList.add('alert-danger');
            statusDiv.textContent = 'Fetch Error: ' + error.message;
             // Show placeholder on fetch error
             if (sys2RequirementsPlaceholder) sys2RequirementsPlaceholder.style.display = 'block';
        });
    }

    // Function to display SYS.2 requirements
    function displaySys2Requirements(requirements) {
        const container = document.getElementById('sys2RequirementsTableContainer');
        const placeholder = document.getElementById('sys2RequirementsPlaceholder');
        const table = document.getElementById('sys2RequirementsTable');
        const tbody = table.querySelector('tbody');

        // Define a character limit for collapsing
        const collapseThreshold = 150; // Adjust as needed

        // Clear previous results
        tbody.innerHTML = '';

        if (!requirements || requirements.length === 0) {
            placeholder.style.display = 'block';
            table.style.display = 'none';
            return;
        }

        placeholder.style.display = 'none';
        table.style.display = 'table';

        requirements.forEach(req => {
            const row = tbody.insertRow();
            row.dataset.sys2Id = req.sys2_id; // Store SYS.2 ID on the row

            // Helper function to create a cell with collapsible text
            function createCollapsibleCell(text) {
                const cell = row.insertCell();
                const textContent = text || '';
                
                if (textContent.length > collapseThreshold) {
                    const containerDiv = document.createElement('div');
                    containerDiv.classList.add('collapsible-text-container');
                    containerDiv.textContent = textContent;

                    const toggleLink = document.createElement('a');
                    toggleLink.href = '#';
                    toggleLink.classList.add('show-more-toggle');
                    toggleLink.textContent = 'Show more';

                    toggleLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        containerDiv.classList.toggle('expanded');
                        if (containerDiv.classList.contains('expanded')) {
                            toggleLink.textContent = 'Show less';
                        } else {
                            toggleLink.textContent = 'Show more';
                        }
                    });

                    cell.appendChild(containerDiv);
                    cell.appendChild(toggleLink);
                } else {
                    cell.textContent = textContent;
                }
                return cell;
            }

            // Cells - ensure order matches table headers and use helper for long text columns
            row.insertCell(0).textContent = req.sys1_id || '';
            createCollapsibleCell(req.sys1_requirement).dataset.field = 'sys1_requirement'; // SYS.1 Requirement - Make collapsible
            row.insertCell(2).textContent = req.sys2_id || '';

            // Make SYS.2 System Requirement editable and potentially collapsible (handled by edit logic)
            const sys2ReqCell = createCollapsibleCell(req.sys2_requirement);
            sys2ReqCell.dataset.field = 'sys2_requirement'; // Mark for editing
            // Note: Edit logic will need adjustment to handle the wrapper div
            row.appendChild(sys2ReqCell); // Append the created cell to the row

            row.insertCell(4).textContent = req.classification || '';
            row.insertCell(5).textContent = req.verification_mapping || '';

            createCollapsibleCell(req.verification_criteria).dataset.field = 'verification_criteria'; // Verification Criteria - Make collapsible

            row.insertCell(7).textContent = req.domain || '';

            // Rationale (originally 9, now 8) - Make collapsible
            createCollapsibleCell(req.rationale).dataset.field = 'rationale'; // Rationale - Make collapsible

            // Requirement Status (originally 10, now 9)
            row.insertCell(9).textContent = req.req_status || '';

            // Actions Cell
            const actionsCell = row.insertCell(10);

            // Initially show only Edit button
            actionsCell.innerHTML = `
                <button class="btn btn-sm btn-primary edit-btn">Edit</button>
                <button class="btn btn-sm btn-success save-btn" style="display: none;">Save</button>
                <button class="btn btn-sm btn-secondary cancel-btn" style="display: none;">Cancel</button>
            `;

            // Add event listeners to buttons
            actionsCell.querySelector('.edit-btn').addEventListener('click', handleEditClick);
            actionsCell.querySelector('.save-btn').addEventListener('click', handleSaveClick);
            actionsCell.querySelector('.cancel-btn').addEventListener('click', handleCancelClick);
        });
    }

    // Store original requirement data for cancelling edits
    let originalRequirementData = {};

    // Handle Edit button click
    function handleEditClick(event) {
        console.log('Edit button clicked', event.target);
        const row = event.target.closest('tr');
        const sys2Id = row.dataset.sys2Id;
        console.log('Editing requirement with SYS.2 ID:', sys2Id);

        // Store original data before editing
        originalRequirementData[sys2Id] = {
            // Target the collapsible text container if it exists, otherwise the cell text
            sys2_requirement: row.cells[3].querySelector('.collapsible-text-container')?.textContent.trim() || row.cells[3].textContent.trim(),
            // Rationale is now at index 8 - only store text, not editable currently
            rationale: row.cells[8].querySelector('.collapsible-text-container')?.textContent.trim() || row.cells[8].textContent.trim()
            // Requirement Status is now at index 9
            // req_status: row.cells[9].textContent.trim() // Status is not editable per current UI
            // Add other editable fields here if any
        };
        console.log('Stored original data:', originalRequirementData[sys2Id]);

        // Enable editing for relevant cells
        const sys2ReqCellContent = row.cells[3].querySelector('.collapsible-text-container') || row.cells[3]; // Get the container or the cell
        sys2ReqCellContent.contentEditable = true; // SYS.2 System Requirement
        sys2ReqCellContent.classList.add('editable'); // Add a class for styling if needed

        // Hide the toggle link if it exists during editing
        const sys2ReqToggle = row.cells[3].querySelector('.show-more-toggle');
        if (sys2ReqToggle) {
            sys2ReqToggle.style.display = 'none';
        }


        // Show Save/Cancel buttons, hide Edit button (Deps/Feedback already hidden)
        row.querySelector('.edit-btn').style.display = 'none';
        row.querySelector('.save-btn').style.display = 'inline-block';
        row.querySelector('.cancel-btn').style.display = 'inline-block';
    }

    // Handle Save button click
    function handleSaveClick(event) {
        console.log('Save button clicked', event.target);
        const row = event.target.closest('tr');
        const sys2Id = row.dataset.sys2Id;
        console.log('Saving requirement with SYS.2 ID:', sys2Id);

        // Collect updated data
        const sys2ReqCellContent = row.cells[3].querySelector('.collapsible-text-container') || row.cells[3]; // Get the container or the cell

        const updatedData = {
            sys2_requirement: sys2ReqCellContent.textContent.trim(),
            // Rationale is now at index 8 - not editable currently
            rationale: row.cells[8].querySelector('.collapsible-text-container')?.textContent.trim() || row.cells[8].textContent.trim()
            // Requirement Status is now at index 9
            // req_status: row.cells[9].textContent.trim() // Status is not editable per current UI
            // Collect other editable fields here if any
        };
         console.log('Collected updated data:', updatedData);

        // Send update to backend
        updateRequirement(sys2Id, updatedData);

        // Disable editing and restore button states
        disableEditing(row);
    }

    // Handle Cancel button click
    function handleCancelClick(event) {
        console.log('Cancel button clicked', event.target);
        const row = event.target.closest('tr');
        const sys2Id = row.dataset.sys2Id;
        console.log('Cancelling edit for requirement with SYS.2 ID:', sys2Id);

        // Restore original data
        if (originalRequirementData[sys2Id]) {
            const sys2ReqCellContent = row.cells[3].querySelector('.collapsible-text-container') || row.cells[3]; // Get the container or the cell
            sys2ReqCellContent.textContent = originalRequirementData[sys2Id].sys2_requirement;
            // Rationale is now at index 8
            // row.cells[8].textContent = originalRequirementData[sys2Id].rationale; // Rationale is not editable
             // Restore other editable fields if any

            // Remove original data from storage
            delete originalRequirementData[sys2Id];
        }

        // Disable editing and restore button states
        disableEditing(row);
    }

    // Helper function to disable editing and reset button states
    function disableEditing(row) {
        console.log('Disabling editing for row:', row);

        const sys2ReqCellContent = row.cells[3].querySelector('.collapsible-text-container') || row.cells[3]; // Get the container or the cell
        sys2ReqCellContent.contentEditable = false;
        sys2ReqCellContent.classList.remove('editable');

         // Show the toggle link again if it exists
        const sys2ReqToggle = row.cells[3].querySelector('.show-more-toggle');
        if (sys2ReqToggle) {
            sys2ReqToggle.style.display = 'block';
             // Ensure the container is collapsed when editing finishes
            sys2ReqCellContent.classList.remove('expanded');
            // Update toggle link text back to 'Show more'
            sys2ReqToggle.textContent = 'Show more';
        }


        row.querySelector('.edit-btn').style.display = 'inline-block';
        row.querySelector('.save-btn').style.display = 'none';
        row.querySelector('.cancel-btn').style.display = 'none';
    }

    // Function to send update request to the backend (Already partially implemented)
    async function updateRequirement(sys2Id, updates) {
        console.log('Sending update request for SYS.2 ID:', sys2Id, 'with updates:', updates);
        // Ensure this matches the endpoint in app.py
        const url = `/api/agent2/update_requirement`; 

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    id: sys2Id, // Use 'id' as expected by the backend
                    updates: updates,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to update requirement');
            }

            const result = await response.json();
            console.log('Update successful:', result);
            // Optionally, show a success message to the user

        } catch (error) {
            console.error('Error updating requirement:', error);
            // Optionally, show an error message to the user
            // You might want to revert the changes in the UI if the save fails
            // This would require storing the original state more robustly or refetching.
        }
    }

    // Function to export requirements
    function exportRequirements(format) {
        console.log(`Exporting SYS.2 requirements in ${format} format.`);
        fetch(`/api/agent2/export/${format}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Export failed');
                }
                return response.blob();
            })
            .then(blob => {
                // Create a temporary URL for the blob
                const url = window.URL.createObjectURL(blob);
                
                // Create a temporary link element
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `sys2_requirements.${format}`;
                
                // Append to body, click, and remove
                document.body.appendChild(a);
                a.click();
                
                // Clean up
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                // Show success toast
                const toastEl = document.getElementById('exportToast');
                const toast = new bootstrap.Toast(toastEl, {
                    delay: 3000 // Show for 3 seconds
                });
                toastEl.querySelector('.toast-body').textContent = 'Export completed successfully!';
                toast.show();
            })
            .catch(error => {
                console.error('Error exporting:', error);
                // Show error toast
                const toastEl = document.getElementById('exportToast');
                const toast = new bootstrap.Toast(toastEl, {
                    delay: 3000 // Show for 3 seconds
                });
                toastEl.querySelector('.toast-body').textContent = 'Export failed: ' + error.message;
                toast.show();
            });
    }

    // Function to show dependencies in modal
    function showDependencies(sys2Id, dependencies) {
        const modal = new bootstrap.Modal(document.getElementById('dependenciesModal'));
        const graphContainer = document.getElementById('dependenciesGraph');
        const listContainer = document.getElementById('dependenciesListItems');
        
        // Clear previous content
        listContainer.innerHTML = '';
        
        // Create nodes and edges for the graph
        const nodes = new vis.DataSet([
            { id: sys2Id, label: sys2Id, color: '#28a745' } // Current requirement
        ]);
        const edges = new vis.DataSet();
        
        // Add dependencies to the graph and list
        dependencies.forEach(dep => {
            nodes.add({ id: dep.to, label: dep.to });
            edges.add({ from: dep.from, to: dep.to, label: dep.label });
            
            // Add to list
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = `${dep.from} → ${dep.to} (${dep.label})`;
            listContainer.appendChild(li);
        });
        
        // Create the network
        const data = { nodes, edges };
        const options = {
            nodes: {
                shape: 'box',
                margin: 10,
                font: { size: 14 }
            },
            edges: {
                arrows: 'to',
                font: { size: 12 }
            },
            layout: {
                hierarchical: {
                    direction: 'LR',
                    sortMethod: 'directed'
                }
            }
        };
        
        new vis.Network(graphContainer, data, options);
        modal.show();
    }

    // Function to show Agent 3 feedback
    function showAgent3Feedback(sys2Id) {
        const modal = new bootstrap.Modal(document.getElementById('agent3FeedbackModal'));
        const feedbackContent = document.getElementById('feedbackContent');
        const suggestionsList = document.getElementById('feedbackSuggestions');
        const complianceResults = document.getElementById('complianceResults');
        const testProposals = document.getElementById('testProposals');
        const improvementSuggestion = document.getElementById('improvementSuggestion');
        const submitImprovement = document.getElementById('submitImprovement');
        
        // Clear previous content
        feedbackContent.innerHTML = '';
        suggestionsList.innerHTML = '';
        complianceResults.innerHTML = '';
        testProposals.innerHTML = '';
        improvementSuggestion.value = '';
        
        // Fetch feedback from Agent 3
        fetch(`/api/agent3/review/${sys2Id}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Display review status
                    feedbackContent.innerHTML = `
                        <div class="alert alert-info">
                            <h6>Review Status</h6>
                            <p>${data.review_status || 'Pending review'}</p>
                        </div>
                    `;
                    
                    // Display compliance results
                    if (data.compliance_results && data.compliance_results.length > 0) {
                        const complianceHtml = data.compliance_results.map(result => `
                            <div class="alert ${result.status === 'Pass' ? 'alert-success' : 'alert-warning'}">
                                <strong>${result.rule}</strong>: ${result.description}
                            </div>
                        `).join('');
                        complianceResults.innerHTML = complianceHtml;
                    } else {
                        complianceResults.innerHTML = '<p class="text-muted">No compliance results available</p>';
                    }
                    
                    // Display suggestions
                    if (data.suggestions && data.suggestions.length > 0) {
                        data.suggestions.forEach(suggestion => {
                            const li = document.createElement('li');
                            li.className = 'list-group-item';
                            li.textContent = suggestion;
                            suggestionsList.appendChild(li);
                        });
                    } else {
                        suggestionsList.innerHTML = '<li class="list-group-item">No suggestions available</li>';
                    }
                    
                    // Display test proposals
                    if (data.test_proposals && data.test_proposals.length > 0) {
                        data.test_proposals.forEach(proposal => {
                            const li = document.createElement('li');
                            li.className = 'list-group-item';
                            li.innerHTML = `
                                <strong>${proposal.title}</strong>
                                <p class="mb-0">${proposal.description}</p>
                            `;
                            testProposals.appendChild(li);
                        });
                    } else {
                        testProposals.innerHTML = '<li class="list-group-item">No test proposals available</li>';
                    }
                    
                    // Set up submit improvement handler
                    submitImprovement.onclick = () => {
                        const suggestion = improvementSuggestion.value.trim();
                        if (!suggestion) {
                            alert('Please enter a suggestion');
                            return;
                        }
                        
                        // Submit the suggestion
                        fetch('/api/agent3/apply-suggestion', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                requirement_id: sys2Id,
                                suggestion: suggestion
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                alert('Suggestion submitted successfully');
                                improvementSuggestion.value = '';
                                // Optionally refresh the feedback
                                showAgent3Feedback(sys2Id);
                            } else {
                                alert('Error submitting suggestion: ' + data.message);
                            }
                        })
                        .catch(error => {
                            alert('Error submitting suggestion: ' + error.message);
                        });
                    };
                } else {
                    feedbackContent.innerHTML = `
                        <div class="alert alert-warning">
                            <p>No feedback available yet. This requirement is pending review.</p>
                        </div>
                    `;
                }
                modal.show();
            })
            .catch(error => {
                feedbackContent.innerHTML = `
                    <div class="alert alert-danger">
                        <p>Error fetching feedback: ${error.message}</p>
                    </div>
                `;
                modal.show();
            });
    }

    // Dark Mode Toggle Logic (Keep this separate or merge carefully)
    const darkModeToggle = document.getElementById('darkModeToggle');
    const body = document.body;

    // Load saved preference
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode) {
        body.classList.toggle('dark-mode', savedMode === 'enabled');
        if (savedMode === 'enabled') {
            darkModeToggle.checked = true;
        }
    }

    darkModeToggle.addEventListener('change', function() {
        if (this.checked) {
            body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'enabled');
        } else {
            body.classList.remove('dark-mode');
            localStorage.setItem('darkMode', 'disabled');
        }
    });

}); // End DOMContentLoaded

// --- Combined Dashboard Summary & Chart Rendering ---
// Function to fetch and display combined dashboard summary
async function fetchAndDisplayCombinedDashboardSummary() {
    console.log("Fetching combined dashboard summary...");
    try {
        const response = await fetch('/api/agent2/combined_dashboard_summary');
        const data = await response.json();

        if (data.status === 'success') {
            console.log("Combined Dashboard summary fetched successfully:", data.summary);
            const sys1Summary = data.summary?.sys1 || {}; // Keep sys1Summary variable for potential future use/traceability table
            const sys2Summary = data.summary?.sys2 || {};

            // Add a small delay before rendering charts and table
            setTimeout(() => {
                // Update SYS.2 summary statistics text (using new chart-specific IDs)
                // Ensure all IDs here match the HTML span elements added in the last step
                const totalSys2ReqsChartElement = document.getElementById('totalSys2ReqsChart');
                if (totalSys2ReqsChartElement) totalSys2ReqsChartElement.textContent = sys2Summary.total_sys2_reqs || 0;

                const totalSys2ReqsStatusChartElement = document.getElementById('totalSys2ReqsStatusChart');
                if (totalSys2ReqsStatusChartElement) totalSys2ReqsStatusChartElement.textContent = sys2Summary.total_sys2_reqs || 0;

                const totalSys2ReqsClassificationChartElement = document.getElementById('totalSys2ReqsClassificationChart');
                if (totalSys2ReqsClassificationChartElement) totalSys2ReqsClassificationChartElement.textContent = sys2Summary.total_sys2_reqs || 0;

                 const totalSys2ReqsVerificationChartElement = document.getElementById('totalSys2ReqsVerificationChart');
                 if (totalSys2ReqsVerificationChartElement) totalSys2ReqsVerificationChartElement.textContent = sys2Summary.total_sys2_reqs || 0;

                // Update specific count spans for Overall Status
                const tracedToSys1Element = document.getElementById('tracedToSys1');
                 if (tracedToSys1Element) tracedToSys1Element.textContent = sys2Summary.traced_to_sys1 || 0;

                const notTracedToSys1Element = document.getElementById('notTracedToSys1');
                 if (notTracedToSys1Element) notTracedToSys1Element.textContent = sys2Summary.not_traced_to_sys1 || 0;

                const draftSys2CountElement = document.getElementById('draftSys2Count');
                if (draftSys2CountElement) draftSys2CountElement.textContent = sys2Summary.status_breakdown?.Draft || 0;

                const reviewedSys2CountElement = document.getElementById('reviewedSys2Count');
                if (reviewedSys2CountElement) reviewedSys2CountElement.textContent = sys2Summary.status_breakdown?.Reviewed || 0;

                const approvedSys2CountElement = document.getElementById('approvedSys2Count');
                if (approvedSys2CountElement) approvedSys2CountElement.textContent = sys2Summary.status_breakdown?.Approved || 0;

                const rejectedSys2CountElement = document.getElementById('rejectedSys2Count');
                if (rejectedSys2CountElement) rejectedSys2CountElement.textContent = sys2Summary.status_breakdown?.Rejected || 0;

                const sys2OtherStatusCountElement = document.getElementById('otherSys2Count');
                if(sys2OtherStatusCountElement) sys2OtherStatusCountElement.textContent = sys2Summary.status_breakdown?.Other || 0;

                // Update specific count spans for Classification Breakdown
                const functionalSys2CountElement = document.getElementById('functionalSys2Count');
                 if(functionalSys2CountElement) functionalSys2CountElement.textContent = sys2Summary.classification_breakdown?.Functional || 0;

                const nonFunctionalSys2CountElement = document.getElementById('nonFunctionalSys2Count');
                 if(nonFunctionalSys2CountElement) nonFunctionalSys2CountElement.textContent = sys2Summary.classification_breakdown?.Non_Functional || 0;

                 const assumptionSys2CountElement = document.getElementById('assumptionSys2Count');
                 if(assumptionSys2CountElement) assumptionSys2CountElement.textContent = sys2Summary.classification_breakdown?.Assumption || 0;

                 const constraintSys2CountElement = document.getElementById('constraintSys2Count');
                 if(constraintSys2CountElement) constraintSys2CountElement.textContent = sys2Summary.classification_breakdown?.Constraint || 0;

                const sys2OtherTypeCountElement = document.getElementById('otherSys2TypeCount');
                if(sys2OtherTypeCountElement) sys2OtherTypeCountElement.textContent = sys2Summary.classification_breakdown?.Other || 0;

                // Update specific count spans for Verification Level Breakdown
                const sqtSys2CountElement = document.getElementById('sqtSys2Count');
                if(sqtSys2CountElement) sqtSys2CountElement.textContent = sys2Summary.verification_breakdown?.['System Qualification Test (SYS.5)'] || 0;

                const sitSys2CountElement = document.getElementById('sitSys2Count');
                if(sitSys2CountElement) sitSys2CountElement.textContent = sys2Summary.verification_breakdown?.['System Integration Test (SYS.4)'] || 0;

                const analysisSys2CountElement = document.getElementById('analysisSys2Count');
                if(analysisSys2CountElement) analysisSys2CountElement.textContent = sys2Summary.verification_breakdown?.Analysis || 0;

                const inspectionSys2CountElement = document.getElementById('inspectionSys2Count');
                if(inspectionSys2CountElement) inspectionSys2CountElement.textContent = sys2Summary.verification_breakdown?.Inspection || 0;

                const sys2OtherVerificationCountElement = document.getElementById('otherVerificationCount');
                if(sys2OtherVerificationCountElement) sys2OtherVerificationCountElement.textContent = sys2Summary.verification_breakdown?.Other || 0;

               // Render Charts (using new chart IDs)
               renderChart('sys2TraceabilityChart', ['Traced to SYS.1', 'Not Traced to SYS.1'], [sys2Summary.traced_to_sys1 || 0, sys2Summary.not_traced_to_sys1 || 0], ['#007bff', '#ffc107']);
               renderChart('sys2StatusChart', Object.keys(sys2Summary.status_breakdown || {}), Object.values(sys2Summary.status_breakdown || {}), ['#ffc107', '#17a2b8', '#28a745', '#dc3545', '#6c757d']); // Yellow (Draft), Info (Reviewed), Green (Approved), Red (Rejected), Secondary (Other)
               renderChart('sys2TypeChart', Object.keys(sys2Summary.classification_breakdown || {}), Object.values(sys2Summary.classification_breakdown || {}), ['#28a745', '#007bff', '#ffc107', '#dc3545', '#6c757d']); // Green (Functional), Blue (Non-Functional), Yellow (Assumption), Red (Constraint), Secondary (Other)
               renderChart('sys2VerificationChart', Object.keys(sys2Summary.verification_breakdown || {}), Object.values(sys2Summary.verification_breakdown || {}), ['#17a2b8', '#007bff', '#28a745', '#ffc107', '#6c757d', '#dc3545']); // Info (SQT), Blue (Test), Green (Analysis), Yellow (Inspection), Secondary (Other), Red (Default/Unexpected)

               // Display Traceability Table
               displaySys2TraceabilityTable(data.summary?.sys1_requirements || [], data.summary?.sys2_requirements || []);

           }, 100); // 100ms delay

        } else {
            console.error("Error fetching combined dashboard summary:", data.message);
        }
    } catch (error) {
        console.error("Fetch error for combined dashboard summary:", error);
    }
}

// Generic function to render a pie chart
let chartInstances = {}; // Store chart instances by id
function renderChart(canvasId, labels, data, colors) {
   const ctx = document.getElementById(canvasId)?.getContext('2d');
   if (!ctx) {
       console.error(`Canvas element with ID ${canvasId} not found.`);
       return;
   }

   // Destroy existing chart if it exists
   if (chartInstances[canvasId]) {
       chartInstances[canvasId].destroy();
   }

   chartInstances[canvasId] = new Chart(ctx, {
       type: 'pie',
       data: {
           labels: labels,
           datasets: [{
               data: data,
               backgroundColor: colors,
           }]
       },
       options: {
           responsive: true,
           plugins: {
               legend: {
                   position: 'top',
                    labels: {
                        color: document.body.classList.contains('dark-mode') ? '#f8f9fa' : '#212529' // Adjust legend text color based on mode
                    }
               },
               tooltip: {
                   callbacks: {
                       label: function(tooltipItem) {
                           return tooltipItem.label + ': ' + tooltipItem.raw;
                       }
                   }
               }
           }
       },
   });
    // Add a handler for dark mode toggle to update chart legend color
    document.getElementById('darkModeToggle')?.addEventListener('change', () => {
        if (chartInstances[canvasId] && chartInstances[canvasId].options.plugins.legend.labels) {
             chartInstances[canvasId].options.plugins.legend.labels.color = document.body.classList.contains('dark-mode') ? '#f8f9fa' : '#212529';
             chartInstances[canvasId].update();
        }
    });
}

// Function to display SYS.1 to SYS.2 traceability table
function displaySys2TraceabilityTable(sys1Reqs, sys2Reqs) {
   console.log("TRACE_TABLE: Attempting to display SYS.1 to SYS.2 traceability table.");
   console.log("TRACE_TABLE: SYS.1 Req Data Received:", sys1Reqs);
   console.log("TRACE_TABLE: SYS.2 Req Data Received:", sys2Reqs);

   const table = document.getElementById('sys2TraceabilityTable');
   const tableBody = table ? table.querySelector('tbody') : null;
   const placeholder = document.getElementById('sys2TraceabilityPlaceholder');

   if (!table || !tableBody || !placeholder) {
       console.error("Traceability table elements not found.");
       return;
   }

   tableBody.innerHTML = ''; // Clear existing rows

   if (!sys1Reqs || sys1Reqs.length === 0 || !sys2Reqs || sys2Reqs.length === 0) {
       placeholder.style.display = 'block';
       table.style.display = 'none';
       console.log("No data for traceability table, showing placeholder.");
       return;
   }

   placeholder.style.display = 'none';
   table.style.display = 'table'; // Show the table

   // Create a map of SYS.1 requirements for easier lookup by sys1_id
   const sys1Map = {};
   sys1Reqs.forEach(req => {
       if (req.sys1_id) {
           sys1Map[req.sys1_id] = req;
       }
   });

   // Iterate through SYS.2 requirements and find their corresponding SYS.1 requirements
   sys2Reqs.forEach(sys2Req => {
       // Assuming sys2Req has a field linking it back to SYS.1, e.g., 'sys1_id'
       const sys1Id = sys2Req.sys1_id;
       console.log(`TRACE_TABLE: Processing SYS.2 Req ID: ${sys2Req.sys2_id}, Linked SYS.1 ID: ${sys1Id}`);

       const row = tableBody.insertRow();
       const sys1Cell = row.insertCell(0);
       const sys2Cell = row.insertCell(1);

       sys2Cell.textContent = sys2Req.sys2_id || '';

       if (sys1Id) {
           // Find the corresponding SYS.1 requirement (optional - just displaying ID for now)
           // const sys1Req = sys1Map[sys1Id];
           sys1Cell.textContent = sys1Id;
           console.log(`TRACE_TABLE: Added row for SYS.1 ${sys1Id} and SYS.2 ${sys2Req.sys2_id}`);
       } else {
           // Handle SYS.2 requirements that are not traced to any SYS.1 requirement
           sys1Cell.textContent = 'Untraced';
           console.log(`TRACE_TABLE: Added row for untraced SYS.2 ${sys2Req.sys2_id}`);
       }
   });
    console.log("TRACE_TABLE: Traceability table population finished.");
}

// Fetch and display combined summary on page load
document.addEventListener('DOMContentLoaded', fetchAndDisplayCombinedDashboardSummary);

// Modify the handleProcessResponse function to also update the dashboard
// Find the existing handleProcessResponse function
const originalHandleProcessResponse = window.handleProcessResponse;
window.handleProcessResponse = function(data) {
   // Call the original function first
   originalHandleProcessResponse(data);

   // Then fetch and display the updated dashboard summary
   fetchAndDisplayCombinedDashboardSummary();
};
// --- End Combined Dashboard Summary & Chart Rendering ---

// Function to automatically export SYS.2 requirements to XLSX and show a toast
function autoExportSys2Requirements() {
    fetch('/api/agent2/export/xlsx')
        .then(response => {
            if (!response.ok) throw new Error('Export failed');
            return response.blob();
        })
        .then(blob => {
            // Download the file
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'sys2_requirements.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            // Show success toast
            const toastEl = document.getElementById('exportToast');
            if (toastEl) {
                const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
                toastEl.querySelector('.toast-body').textContent = 'Export completed successfully!';
                toast.show();
            }
        })
        .catch(error => {
            // Show error toast
            const toastEl = document.getElementById('exportToast');
            if (toastEl) {
                const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
                toastEl.querySelector('.toast-body').textContent = 'Export failed: ' + error.message;
                toast.show();
            }
        });
} 