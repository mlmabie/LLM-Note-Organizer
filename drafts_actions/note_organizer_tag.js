// Note Organizer: Tag with AI
// This action sends the current draft to the Note Organizer service
// and asks for tag suggestions. It presents these to the user for selection,
// then adds the selected tags to the draft.

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
 * Main function
 */
function main() {
    try {
        // Get the current draft content
        const content = draft.content;
        const title = draft.title;
        
        // Confirm with user
        if (!confirm("AI Tag Suggestion", 
            "This will analyze your note and suggest tags using AI. Continue?")) {
            return;
        }
        
        // Show processing message
        app.displayInfoMessage("Analyzing note...");
        
        // Prepare request
        const requestData = {
            content: content,
            action: "tag",
            part: "entire",
            options: {}
        };
        
        // Call API
        let result = callApi("process", requestData);
        
        if (!result.success) {
            showError(`Failed to process note: ${result.message}`);
            return;
        }
        
        // If we got tag suggestions, show them to the user
        if (result.suggested_tags && result.suggested_tags.length > 0) {
            // Format tags with confidence scores
            const tagOptions = result.suggested_tags.map(tag => {
                const confidence = Math.round(tag.confidence * 100);
                return `${tag.name} (${confidence}%)`;
            });
            
            // Show tag selection prompt
            let p = new Prompt();
            p.title = "Select Tags to Add";
            p.message = "The following tags were suggested for your note. Select the ones you want to add:";
            p.addSelect("tags", "Tags", tagOptions, [], true);
            p.addButton("Add Selected Tags");
            p.addButton("Cancel");
            
            if (p.show() === 0) { // User pressed "Add Selected Tags"
                const selectedIndices = p.fieldValues["tags"];
                const selectedTags = selectedIndices.map(i => result.suggested_tags[i].name);
                
                // Add the selected tags to the draft
                for (let tag of selectedTags) {
                    draft.addTag(tag);
                }
                
                // Update draft
                draft.update();
                
                // Show success message
                app.displaySuccessMessage(`Added ${selectedTags.length} tags to draft`);
            }
        } else {
            app.displayInfoMessage("No tags suggested. Try a longer or more detailed note.");
        }
    } catch (error) {
        console.error(error);
        showError(`An error occurred: ${error.message}`);
    }
}

// Run the main function
main(); 