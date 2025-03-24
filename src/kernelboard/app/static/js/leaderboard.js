// Toggles the visibility of the reference implementation code block.
document.addEventListener('DOMContentLoaded', function() {
    const referenceImpl = document.getElementById('referenceImpl');
    const toggleBtn = document.getElementById('toggleCodeBtn');

    let isExpanded = false;
    
    if (toggleBtn && referenceImpl) {
        const codeBlockFade = document.createElement('div');
        codeBlockFade.className = 'code-block-fade';
        referenceImpl.appendChild(codeBlockFade);

        toggleBtn.addEventListener('click', function() {
            if (isExpanded) {
                referenceImpl.classList.remove('max-h-none');
                referenceImpl.classList.add('max-h-[300px]');
                referenceImpl.classList.remove('overflow-y-auto');
                referenceImpl.classList.add('overflow-y-hidden');
                toggleBtn.textContent = 'Show';
                codeBlockFade.style.display = 'block';
                isExpanded = false;
            } else {
                referenceImpl.classList.remove('max-h-[300px]');
                referenceImpl.classList.add('max-h-none');
                referenceImpl.classList.remove('overflow-y-hidden');
                referenceImpl.classList.add('overflow-y-auto');
                toggleBtn.textContent = 'Hide';
                codeBlockFade.style.display = 'none';
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
    const codeElement = document.querySelector('#codeBlock');
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