// Toggles the visibility of the reference implementation code block.
document.addEventListener('DOMContentLoaded', function() {
    const codeBlock = document.getElementById('codeBlock');
    const toggleBtn = document.getElementById('toggleCodeBtn');
    const gradientOverlay = document.getElementById('gradientOverlay');
    
    let isExpanded = false;
    
    if (toggleBtn && codeBlock) {
        toggleBtn.addEventListener('click', function() {
            if (isExpanded) {
                codeBlock.classList.remove('code-block-hidden');
                codeBlock.classList.add('max-h-[200px]');
                codeBlock.classList.remove('overflow-y-auto');
                codeBlock.classList.add('overflow-y-hidden');
                toggleBtn.textContent = 'Show';
                gradientOverlay.classList.remove('hidden');
                isExpanded = false;
            } else {
                codeBlock.classList.remove('max-h-[200px]');
                codeBlock.classList.add('max-h-none');
                codeBlock.classList.remove('overflow-y-hidden');
                codeBlock.classList.add('overflow-y-auto');
                toggleBtn.textContent = 'Hide';
                gradientOverlay.classList.add('hidden');
                isExpanded = true;
            }
        });
    }
});

/**
 * Copies the code from the reference implementation code block to the
 * clipboard.
 */
function copyCode() {
    const codeElement = document.querySelector('.code-block code');
    const textArea = document.createElement('textarea');
    textArea.value = codeElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    const button = document.querySelector('button[onclick="copyCode()"]');
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
} 

/**
 * Toggles the visibility of the rankings for a given GPU type.
 * @param {string} gpuId - The ID of the GPU type to toggle rankings for.
 * @param {number} totalCount - The total number of rankings to display.
 */
function toggleRankings(gpuId, totalCount) {
    const section = document.getElementById('section-' + gpuId);
    const button = section.querySelector('.rankings-btn');
    const rows = section.querySelectorAll('tr.rank-row');
    
    // Check if we're currently showing all (button text contains "Show Top 3")
    const isShowingAll = button.textContent.includes('Show Top 3');
    
    // Toggle visibility based on current state
    rows.forEach(row => {
        const rank = parseInt(row.getAttribute('data-rank'));
        if (rank > 3) {
            if (isShowingAll) {
                // Hide rows beyond top 3
                row.classList.add('hidden-row');
            } else {
                // Show all rows
                row.classList.remove('hidden-row');
            }
        }
    });
    
    // Update button text
    if (isShowingAll) {
        button.textContent = `Show All (${totalCount})`;
    } else {
        button.textContent = 'Show Top 3';
    }
}