# Note Organizer Drafts Integration

This folder contains a set of Drafts actions that integrate with the Note Organizer system, allowing for AI-powered note management directly from the Drafts app.

## Features

- **AI-powered tagging**: Get intelligent tag suggestions for your notes
- **Note refinement**: Improve your writing with AI assistance
- **Smart export to Obsidian**: Export notes to Obsidian with AI-suggested folder organization
- **Structured user interactions**: All actions include clear user prompts and confirmation steps

## Prerequisites

1. [Drafts app](https://getdrafts.com/) for iOS, iPadOS, or macOS
2. A running instance of Note Organizer with API enabled
3. (Optional) [Obsidian](https://obsidian.md/) for note storage

## Installation

### Option 1: Install from Drafts Directory

1. Open the Drafts app
2. Go to "Actions" > "Action Directory"
3. Search for "Note Organizer"
4. Install the actions you want to use

### Option 2: Manual Installation

1. In the Drafts app, go to "Actions" > "Manage Actions"
2. Tap the "+" button to create a new action
3. Tap "Script" in the action editor
4. Copy and paste the JavaScript code from the corresponding `.js` file in this folder
5. Configure any additional settings (name, icon, etc.)
6. Save the action

## Configuration

### API Connection

On first run of any of the actions, you'll be prompted to enter:

1. **API URL**: The URL of your Note Organizer API (e.g., `http://localhost:8000`)
2. **API Key**: Your API key for authentication

### Obsidian Export Configuration

For the Obsidian export action, you'll need to configure:

1. **Vault Path**: The path to your Obsidian vault
2. **AI Folder Suggestion**: Whether to use AI for suggesting folders
3. **Front Matter**: Whether to add YAML front matter to exported notes
4. **Tags**: Whether to include tags from Drafts in the exported note

## Available Actions

### Note Organizer: Tag with AI

This action analyzes your note and suggests relevant tags based on the content. You can select which tags to apply.

**Usage:**
1. Create or open a note in Drafts
2. Run the "Note Organizer: Tag with AI" action
3. Confirm the analysis
4. Select the tags you want to apply from the suggestions
5. The selected tags will be added to your draft

### Note Organizer: Refine with AI

This action helps improve your writing with AI assistance. You can refine the entire note or just a selection.

**Usage:**
1. Create or open a note in Drafts
2. Optionally select a portion of text to refine
3. Run the "Note Organizer: Refine with AI" action
4. Choose refinement options (style, improvements)
5. Preview the refined content
6. Choose how to apply the changes (replace, append, or copy)

### Export to Obsidian with AI Organization

This action exports your note to Obsidian, using AI to suggest the best folder organization.

**Usage:**
1. Create or open a note in Drafts
2. Run the "Export to Obsidian with AI Organization" action
3. Configure export settings
4. Select or confirm the suggested destination folder
5. Provide a filename
6. The note will be exported to your Obsidian vault

## Tips and Best Practices

1. **Provide context**: The more detailed your notes, the better the AI suggestions will be
2. **Review AI suggestions**: Always review AI-generated content before applying changes
3. **Use tags**: Adding relevant tags to your notes helps improve folder suggestions
4. **Organize in batches**: Process multiple notes at once for a more consistent organization

## Troubleshooting

### Action fails to connect to API

- Ensure the Note Organizer API is running
- Verify the API URL and key are correct
- Check for network connectivity issues

### Poor or irrelevant suggestions

- Try using more detailed notes
- Make sure the Note Organizer is properly configured with an LLM provider
- Check the API logs for errors

### Obsidian export fails

- Verify the Obsidian vault path is correct
- Ensure you have write permissions to the vault directory
- Check for illegal characters in filenames

## Contributing

If you have ideas for improving these actions or want to report a bug, please open an issue in the Note Organizer repository.

## License

These actions are released under the same license as the Note Organizer project. 