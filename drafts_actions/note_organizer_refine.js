// Note Organizer: Refine with AI
// This action sends the current draft or selection to the Note Organizer service
// for AI-powered refinement and improvement.

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
 * Gets refinement options from user
 * @returns {object} Refinement options
 */
function getRefineOptions() {
    let p = new Prompt();
    p.title = "Refinement Options";
    p.message = "How would you like to refine this note?";
    
    p.addSelect("style", "Writing Style", [
        "No change",
        "Academic",
        "Professional",
        "Conversational",
        "Concise"
    ], ["No change"], false);
    
    p.addSelect("improve", "Improvements", [
        "Grammar and spelling",
        "Clarity and readability",
        "Structure and organization",
        "All of the above"
    ], ["All of the above"], false);
    
    p.addSwitch("keep_original", "Keep original content", true);
    
    p.addButton("Continue");
    p.addButton("Cancel");
    
    if (p.show() === 0) {
        return {
            style: p.fieldValues["style"],
            improve: p.fieldValues["improve"],
            keep_original: p.fieldValues["keep_original"]
        };
    }
    
    return null;
}

/**
 * Main function
 */
function main() {
    try {
        // Check if there's a selection
        const hasSelection = editor.getSelectedText().length > 0;
        const [selectionStart, selectionLength] = editor.getSelectedRange();
        
        // Get the content to process
        const wholeContent = draft.content;
        const selectedText = hasSelection ? editor.getSelectedText() : wholeContent;
        
        // Confirm with user
        if (!confirm("AI Refinement", 
            `This will refine ${hasSelection ? "the selected text" : "your entire note"} using AI. Continue?`)) {
            return;
        }
        
        // Get refinement options
        const options = getRefineOptions();
        if (!options) {
            return; // User cancelled
        }
        
        // Show processing message
        app.displayInfoMessage("Processing with AI...");
        
        // Prepare request
        const requestData = {
            content: wholeContent,
            action: "refine",
            part: hasSelection ? "selection" : "entire",
            selection_start: hasSelection ? selectionStart : null,
            selection_end: hasSelection ? selectionStart + selectionLength : null,
            options: options
        };
        
        // Call API
        let result = callApi("process", requestData);
        
        if (!result.success) {
            showError(`Failed to process note: ${result.message}`);
            return;
        }
        
        // If we got refined content, show a preview and ask user if they want to apply it
        if (result.content) {
            // Preview the refined content
            let p = new Prompt();
            p.title = "Refined Content";
            p.message = "Here's the refined content. Would you like to apply it?";
            
            p.addTextView("preview", "Preview", result.content);
            
            p.addSelect("action", "Action", [
                "Replace original",
                "Insert after original",
                "Copy to clipboard"
            ], ["Replace original"], false);
            
            p.addButton("Apply");
            p.addButton("Cancel");
            
            if (p.show() === 0) { // User pressed "Apply"
                const action = p.fieldValues["action"];
                
                if (action === "Replace original") {
                    if (hasSelection) {
                        editor.setSelectedText(result.content);
                    } else {
                        draft.content = result.content;
                        draft.update();
                    }
                    app.displaySuccessMessage("Content replaced successfully");
                } 
                else if (action === "Insert after original") {
                    if (hasSelection) {
                        const originalText = editor.getSelectedText();
                        editor.setSelectedText(originalText + "\n\n## Refined Version\n\n" + result.content);
                    } else {
                        draft.content += "\n\n## Refined Version\n\n" + result.content;
                        draft.update();
                    }
                    app.displaySuccessMessage("Refined content added");
                }
                else if (action === "Copy to clipboard") {
                    app.setClipboard(result.content);
                    app.displaySuccessMessage("Copied to clipboard");
                }
            }
        } else {
            app.displayInfoMessage("No refinements were made to your note.");
        }
    } catch (error) {
        console.error(error);
        showError(`An error occurred: ${error.message}`);
    }
}

// Run the main function
main(); 