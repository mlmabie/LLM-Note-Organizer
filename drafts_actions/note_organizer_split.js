// Note Organizer: Split Note
// This action analyzes a long note and suggests how to split it
// into multiple smaller, focused notes with proper organization.

/**
 * Shows a confirmation dialog
 * @param {string} title Dialog title
 * @param {string} message Dialog message
 * @param {string} confirmButton Text for confirm button
 * @param {string} cancelButton Text for cancel button
 * @returns {boolean} true if confirmed, false if cancelled
 */
function confirm(title, message, confirmButton = "OK", cancelButton = "Cancel") {
    let p = new Prompt();
    p.title = title;
    p.message = message;
    p.addButton(confirmButton);
    p.addButton(cancelButton);
    let result = p.show();
    return result === 0; // First button (confirm) returns 0
}

/**
 * Shows an error message
 * @param {string} message Error message to display
 */
function showError(message) {
    let p = new Prompt();
    p.title = "Error";
    p.message = message;
    p.addButton("OK");
    p.show();
}

/**
 * Sends a request to the Note Organizer API
 * @param {string} endpoint API endpoint
 * @param {object} data Request data
 * @returns {object} API response
 */
function callApi(endpoint, data) {
    // Get API settings
    let credential = Credential.create("NoteOrganizerAPI", "Note Organizer API Settings");
    credential.addTextField("apiUrl", "API URL");
    credential.addPasswordField("apiKey", "API Key");
    credential.authorize();
    
    const apiUrl = credential.getValue("apiUrl") || "http://localhost:8000";
    const apiKey = credential.getValue("apiKey") || "";
    
    // Prepare HTTP request
    let http = HTTP.create();
    let response = http.request({
        "url": `${apiUrl}/api/v1/${endpoint}`,
        "method": "POST",
        "data": data,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`
        }
    });
    
    if (response.success) {
        return response.responseData;
    } else {
        throw new Error(`API request to '${endpoint}' failed: ${response.error || 'Status code ' + response.statusCode}`);
    }
}

/**
 * Creates a new draft with content
 * @param {string} title Note title
 * @param {string} content Note content
 * @param {array} tags Tags to add
 * @returns {object} New draft
 */
function createNewDraft(title, content, tags = []) {
    let newDraft = Draft.create();
    newDraft.content = content;
    
    // Apply tags
    for (let tag of tags) {
        newDraft.addTag(tag);
    }
    
    // Always add a tag to indicate this was split from another note
    newDraft.addTag("split-note");
    
    // Update the draft to save it
    newDraft.update();
    
    return newDraft;
}

/**
 * Main function
 */
function main() {
    try {
        // Get the current draft content
        const content = draft.content;
        const wordCount = content.split(/\s+/).length;
        
        // Check if note is long enough to justify splitting
        if (wordCount < 250) {
            showError("This note is too short to split. Consider adding more content first.");
            return;
        }
        
        // Confirm with user
        if (!confirm("Split Note", 
            `This will analyze your note (${wordCount} words) and suggest how to split it into multiple focused notes. Continue?`)) {
            return;
        }
        
        // Show processing message
        app.displayInfoMessage("Analyzing note structure...");
        
        // Prepare request
        const requestData = {
            content: content,
            action: "split",
            part: "entire",
            options: {
                min_section_length: 100,
                preserve_original: true
            }
        };
        
        // Call API
        let result = callApi("process", requestData);
        
        if (!result.success) {
            showError(`Failed to process note: ${result.message}`);
            return;
        }
        
        // If we got suggested actions for splitting
        if (result.actions && result.actions.length > 0) {
            // Filter for create_note actions
            const createActions = result.actions.filter(action => action.type === "create_note");
            
            if (createActions.length === 0) {
                app.displayInfoMessage("No suitable split points found in your note.");
                return;
            }
            
            // Show split options
            let p = new Prompt();
            p.title = "Split Note Suggestions";
            p.message = `AI suggests splitting this note into ${createActions.length} separate notes:`;
            
            // Add options for each suggested note
            for (let i = 0; i < createActions.length; i++) {
                const action = createActions[i];
                p.addSwitch(`split_${i}`, action.title, true);
                p.addTextView(`preview_${i}`, "Preview", action.content.substring(0, 200) + "...");
            }
            
            // Add original note option
            p.addSwitch("keep_original", "Keep original note", true);
            
            p.addButton("Split Now");
            p.addButton("Cancel");
            
            if (p.show() === 0) { // User pressed "Split Now"
                // Create new drafts for selected splits
                let createdCount = 0;
                for (let i = 0; i < createActions.length; i++) {
                    if (p.fieldValues[`split_${i}`]) {
                        const action = createActions[i];
                        createNewDraft(
                            action.title,
                            action.content,
                            action.tags || []
                        );
                        createdCount++;
                    }
                }
                
                // Handle original note
                if (!p.fieldValues["keep_original"]) {
                    // Archive or delete the original
                    draft.addTag("split-source");
                    draft.update();
                    
                    // Option to archive
                    if (confirm("Archive Original", 
                        "Do you want to archive the original note now that it's been split?",
                        "Archive", "Keep")) {
                        draft.archive();
                    }
                }
                
                // Show success message
                app.displaySuccessMessage(`Created ${createdCount} new notes from the split`);
            }
        } else {
            app.displayInfoMessage("AI couldn't identify good ways to split this note. Try a longer or more structured note.");
        }
    } catch (error) {
        console.error(error);
        showError(`An error occurred: ${error.message}`);
    }
}

// Run the main function
main();