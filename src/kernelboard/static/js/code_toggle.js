/**
 * Toggle the code block between expanded and collapsed state.
 * @param {string} defaultHeight - the default height of the code block
 */
function toggleCode(defaultHeight = '200px') {
    const container = document.getElementById('code-container');
    const expandButton = document.getElementById('expand-button');
    const collapseButton = document.getElementById('collapse-button');
    const fadeOverlay = document.getElementById('fade-overlay');
    
    if (container.style.maxHeight === defaultHeight) {
        container.style.maxHeight = container.scrollHeight + 'px';
        expandButton.classList.add('hidden');
        collapseButton.classList.remove('hidden');
        fadeOverlay.classList.add('hidden');
    } else {
        container.style.maxHeight = defaultHeight;
        expandButton.classList.remove('hidden');
        collapseButton.classList.add('hidden');
        fadeOverlay.classList.remove('hidden');
    }
}
