// NYC Health Prediction System - JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const form = document.getElementById('predictionForm');
    const resultsCard = document.getElementById('resultsCard');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoading = predictBtn.querySelector('.btn-loading');
    
    // Set default date to today
    const dateInput = document.getElementById('date');
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;
    dateInput.max = today; // Don't allow future dates
    
    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            date: document.getElementById('date').value,
            borough: document.getElementById('borough').value
        };
        
        // Validate
        if (!formData.date || !formData.borough) {
            alert('Please select both date and borough');
            return;
        }
        
        // Show loading state
        setLoadingState(true);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
                
                // Smooth scroll to results
                setTimeout(() => {
                    resultsCard.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'nearest' 
                    });
                }, 100);
            } else {
                alert('Error: ' + (data.error || 'Could not generate prediction'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        } finally {
            setLoadingState(false);
        }
    });
    
    // Set loading state
    function setLoadingState(loading) {
        if (loading) {
            predictBtn.disabled = true;
            btnText.style.display = 'none';
            btnLoading.style.display = 'inline-flex';
        } else {
            predictBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    }
    
    // Display results
    function displayResults(data) {
        // Update location and date
        document.getElementById('resultBorough').textContent = data.borough;
        document.getElementById('resultDate').textContent = data.date;
        
        // Risk Level
        const riskLevel = data.risk_level;
        const riskCard = document.getElementById('riskCard');
        const riskLevelEl = document.getElementById('riskLevel');
        const riskProbEl = document.getElementById('riskProbability');
        const riskBarFill = document.getElementById('riskBarFill');
        
        riskLevelEl.textContent = riskLevel;
        const riskPercent = Math.round(data.risk_probability * 100);
        riskProbEl.textContent = riskPercent + '%';
        
        // Update risk card styling
        riskCard.classList.remove('low', 'moderate', 'high');
        if (riskLevel === 'High') {
            riskCard.classList.add('high');
            riskLevelEl.style.color = '#ef4444';
        } else if (riskLevel === 'Moderate') {
            riskCard.classList.add('moderate');
            riskLevelEl.style.color = '#f59e0b';
        } else {
            riskCard.classList.add('low');
            riskLevelEl.style.color = '#10b981';
        }
        
        // Animate risk bar
        setTimeout(() => {
            riskBarFill.style.width = riskPercent + '%';
        }, 100);
        
        // Expected Admissions
        const admissions = Math.round(data.expected_admissions);
        document.getElementById('expectedAdmissions').textContent = admissions.toLocaleString();
        
        // Historical Average
        const historicalAvg = data.historical_average;
        const historicalEl = document.getElementById('historicalAverage');
        const comparisonEl = document.getElementById('comparisonValue');
        
        if (historicalAvg !== null && historicalAvg !== undefined) {
            historicalEl.textContent = Math.round(historicalAvg).toLocaleString();
            
            // Calculate comparison
            const diff = admissions - historicalAvg;
            const percentDiff = historicalAvg > 0 ? (diff / historicalAvg * 100) : 0;
            const sign = diff >= 0 ? '+' : '';
            
            comparisonEl.textContent = sign + percentDiff.toFixed(1) + '%';
            comparisonEl.classList.remove('positive', 'negative');
            comparisonEl.classList.add(diff >= 0 ? 'positive' : 'negative');
        } else {
            historicalEl.textContent = 'N/A';
            document.getElementById('comparison').style.display = 'none';
        }
        
        // Confidence
        document.getElementById('confidenceLevel').textContent = data.confidence.split(' (')[0];
        document.getElementById('confidenceDetail').textContent = 
            data.confidence.includes('(') ? 
            data.confidence.match(/\((.*?)\)/)[1] : 
            'Model-based prediction';
        
        // Interpretation
        const interpretation = generateInterpretation(data);
        document.getElementById('interpretationContent').innerHTML = interpretation;
        
        // Show results card with animation
        resultsCard.style.display = 'block';
    }
    
    // Generate interpretation text
    function generateInterpretation(data) {
        const riskLevel = data.risk_level.toLowerCase();
        const admissions = Math.round(data.expected_admissions);
        const borough = data.borough;
        const historical = data.historical_average;
        
        let text = '<p>';
        
        // Risk assessment
        if (riskLevel === 'high') {
            text += `<strong>⚠️ High Risk:</strong> ${borough} is expected to experience elevated hospitalization levels on this date. `;
        } else if (riskLevel === 'moderate') {
            text += `<strong>⚡ Moderate Risk:</strong> ${borough} is expected to see typical hospitalization levels. `;
        } else {
            text += `<strong>✅ Low Risk:</strong> ${borough} is expected to have lower than average hospitalization levels. `;
        }
        
        // Admission count
        text += `We predict approximately <strong>${admissions}</strong> respiratory and asthma-related admissions. `;
        
        // Historical comparison
        if (historical !== null && historical !== undefined) {
            const diff = admissions - historical;
            if (Math.abs(diff) > 5) {
                if (diff > 0) {
                    text += `This is <strong>${Math.abs(Math.round(diff))}</strong> admissions higher than the historical average for similar dates. `;
                } else {
                    text += `This is <strong>${Math.abs(Math.round(diff))}</strong> admissions lower than the historical average for similar dates. `;
                }
            } else {
                text += `This aligns closely with historical patterns for this time of year. `;
            }
        }
        
        // Recommendations
        text += '</p><p style="margin-top: 1rem;"><strong>Healthcare Impact:</strong> ';
        if (riskLevel === 'high') {
            text += 'Hospitals should prepare for increased patient volume. Consider staffing adjustments and ensuring adequate supplies.';
        } else if (riskLevel === 'moderate') {
            text += 'Standard operating procedures should be sufficient. Monitor for any unexpected changes in admission patterns.';
        } else {
            text += 'Expected lower than average volume. This may be a good opportunity for planned maintenance or staff training.';
        }
        text += '</p>';
        
        return text;
    }
    
    // Add some interactivity to info cards
    const infoCards = document.querySelectorAll('.info-card');
    infoCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
    }
});

// Add console message
console.log('%cNYC Health Prediction System', 'color: #3b82f6; font-size: 20px; font-weight: bold;');
console.log('%cDuke University Alternative Data Project 2024', 'color: #64748b; font-size: 12px;');
