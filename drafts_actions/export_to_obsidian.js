// Export to Obsidian with AI Organization
// This action exports the current draft to Obsidian, using AI to suggest
// the best folder organization based on content analysis.

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
 * Gets Obsidian vault location and settings
 * @returns {object} Vault settings
 */
function getObsidianSettings() {
    // Check if we have stored settings
    let fm = FileManager.createLocal();
    let settingsPath = fm.joinPath(fm.tempDirectory, "obsidian_export_settings.json");
    let settings = {};
    
    if (fm.fileExists(settingsPath)) {
        try {
            let settingsContent = fm.readString(settingsPath);
            settings = JSON.parse(settingsContent);
        } catch (e) {
            console.error("Failed to read settings:", e);
        }
    }
    
    // Show settings prompt
    let p = new Prompt();
    p.title = "Obsidian Export Settings";
    p.message = "Configure your Obsidian vault settings:";
    
    p.addTextField("vaultPath", "Vault Path", settings.vaultPath || "");
    p.addSwitch("useAi", "Use AI for folder suggestion", settings.useAi !== false);
    p.addSwitch("addFrontMatter", "Add YAML Front Matter", settings.addFrontMatter !== false);
    p.addSwitch("includeDraftsTag", "Include tags from Drafts", settings.includeDraftsTag !== false);
    
    p.addButton("Continue");
    p.addButton("Cancel");
    
    if (p.show() === 0) {
        // Save settings
        let newSettings = {
            vaultPath: p.fieldValues["vaultPath"],
            useAi: p.fieldValues["useAi"],
            addFrontMatter: p.fieldValues["addFrontMatter"],
            includeDraftsTag: p.fieldValues["includeDraftsTag"]
        };
        
        try {
            fm.writeString(settingsPath, JSON.stringify(newSettings));
        } catch (e) {
            console.error("Failed to save settings:", e);
        }
        
        return newSettings;
    }
    
    return null;
}

/**
 * Suggests a folder for the note using the Note Organizer API
 * @param {string} content Note content
 * @param {array} tags Note tags
 * @returns {string} Suggested folder path
 */
function suggestFolder(content, tags) {
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
        "url": `${apiUrl}/api/v1/suggest_folder`,
        "method": "POST",
        "data": {
            "content": content,
            "tags": tags
        },
        "headers": {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`
        }
    });
    
    if (response.success) {
        return response.responseData.folder || "Inbox";
    } else {
        console.error("API request failed:", response.error || response.statusCode);
        return "Inbox";
    }
}

/**
 * Gets available folders in Obsidian vault
 * @param {string} vaultPath Path to Obsidian vault
 * @returns {array} List of folders
 */
function getObsidianFolders(vaultPath) {
    let fm = FileManager.createLocal();
    let folders = [];
    
    try {
        // Always include root folders
        folders.push(""); // Root
        folders.push("Inbox");
        
        // Function to recursively list folders
        function listFolders(dir, prefix = "") {
            if (!fm.directoryExists(dir)) {
                return;
            }
            
            let items = fm.listContents(dir);
            for (let item of items) {
                let path = fm.joinPath(dir, item);
                if (fm.directoryExists(path)) {
                    let relativePath = prefix ? `${prefix}/${item}` : item;
                    folders.push(relativePath);
                    listFolders(path, relativePath);
                }
            }
        }
        
        listFolders(vaultPath);
        return folders;
    } catch (e) {
        console.error("Failed to list folders:", e);
        return ["", "Inbox"];
    }
}

/**
 * Prepares front matter for the note
 * @param {string} title Note title
 * @param {array} tags Note tags
 * @param {string} folder Folder path
 * @returns {string} YAML front matter
 */
function prepareFrontMatter(title, tags, folder) {
    let frontMatter = "---\n";
    frontMatter += `title: ${title}\n`;
    frontMatter += `date: ${new Date().toISOString().split('T')[0]}\n`;
    
    if (tags && tags.length > 0) {
        frontMatter += "tags:\n";
        for (let tag of tags) {
            frontMatter += `  - ${tag}\n`;
        }
    }
    
    frontMatter += "---\n\n";
    return frontMatter;
}

/**
 * Sanitizes a string for use as a filename
 * @param {string} str String to sanitize
 * @returns {string} Sanitized string
 */
function sanitizeFilename(str) {
    return str.replace(/[/\\?%*:|"<>]/g, '-').trim();
}

/**
 * Main function
 */
function main() {
    try {
        // Get the current draft content and info
        const content = draft.content;
        const tags = draft.tags;
        
        // Generate a title from draft title or first line
        let title = draft.title;
        if (!title) {
            const firstLine = content.split('\n', 1)[0].trim();
            title = firstLine.replace(/^#+\s*/, '').substring(0, 50);
        }
        
        // Get Obsidian settings
        const settings = getObsidianSettings();
        if (!settings) {
            return; // User cancelled
        }
        
        // Validate vault path
        const vaultPath = settings.vaultPath;
        let fm = FileManager.createLocal();
        if (!fm.directoryExists(vaultPath)) {
            showError(`Obsidian vault not found at: ${vaultPath}`);
            return;
        }
        
        // Show processing message
        app.displayInfoMessage("Preparing export...");
        
        // Get folder suggestion if AI is enabled
        let suggestedFolder = "Inbox";
        if (settings.useAi) {
            app.displayInfoMessage("Getting AI folder suggestion...");
            suggestedFolder = suggestFolder(content, tags);
        }
        
        // Get available folders and ask user to select
        const folders = getObsidianFolders(vaultPath);
        
        let p = new Prompt();
        p.title = "Select Destination Folder";
        p.message = "Choose where to save your note in Obsidian:";
        
        // Find the index of the suggested folder
        const suggestedIndex = folders.findIndex(f => f === suggestedFolder);
        const defaultSelection = suggestedIndex >= 0 ? [suggestedIndex] : [0];
        
        p.addSelect("folder", "Folder", folders, defaultSelection, false);
        p.addTextField("filename", "Filename", sanitizeFilename(title) + ".md");
        
        p.addButton("Export");
        p.addButton("Cancel");
        
        if (p.show() === 0) {
            const selectedFolder = p.fieldValues["folder"];
            const filename = p.fieldValues["filename"];
            
            // Make sure filename ends with .md
            const finalFilename = filename.endsWith(".md") ? filename : `${filename}.md`;
            
            // Prepare the full path
            let notePath;
            if (selectedFolder) {
                // Create the folder if it doesn't exist
                const folderPath = fm.joinPath(vaultPath, selectedFolder);
                if (!fm.directoryExists(folderPath)) {
                    fm.createDirectory(folderPath);
                }
                notePath = fm.joinPath(folderPath, finalFilename);
            } else {
                notePath = fm.joinPath(vaultPath, finalFilename);
            }
            
            // Prepare note content
            let noteContent = content;
            
            // Add front matter if needed
            if (settings.addFrontMatter) {
                // Check if the note already has front matter
                if (!noteContent.startsWith("---")) {
                    // Get tags to include
                    const tagsToInclude = settings.includeDraftsTag ? tags : [];
                    
                    // Add front matter
                    const frontMatter = prepareFrontMatter(title, tagsToInclude, selectedFolder);
                    noteContent = frontMatter + noteContent;
                }
            }
            
            // Write the file
            try {
                fm.writeString(notePath, noteContent);
                app.displaySuccessMessage(`Exported to Obsidian: ${selectedFolder}/${finalFilename}`);
                
                // Optionally, we could archive the draft in Drafts or add a tag to indicate it's been exported
                draft.addTag("exported-to-obsidian");
                draft.update();
            } catch (e) {
                showError(`Failed to write file: ${e.message}`);
            }
        }
    } catch (error) {
        console.error(error);
        showError(`An error occurred: ${error.message}`);
    }
}

// Run the main function
main(); 