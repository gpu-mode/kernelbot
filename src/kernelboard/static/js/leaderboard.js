// Function to toggle between showing all rankings and only top 3
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