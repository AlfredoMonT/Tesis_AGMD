document.addEventListener('DOMContentLoaded', function() {
      const form = document.getElementById('assessment-form');
      const analyzeBtn = document.getElementById('analyze-btn');
      const analyzeLoader = document.getElementById('analyze-loader');
      const resetBtn = document.getElementById('reset-btn');
      const resultCard = document.getElementById('risk-result');
      const resultHeader = document.getElementById('result-header');
      const resultIcon = document.getElementById('result-icon');
      const resultTitle = document.getElementById('result-title');
      const resultMessage = document.getElementById('result-message');
      const riskFactors = document.getElementById('risk-factors');
      const recommendationText = document.getElementById('recommendation-text');
      const printBtn = document.getElementById('print-btn');
      
      // Remove this line that loads last assessment from localStorage
      // loadLastAssessment();
      
      // Form submission handler
      form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeLoader.style.display = 'inline-block';
        
        // Simulate API processing
        setTimeout(() => {
          analyzeStudent();
          analyzeBtn.disabled = false;
          analyzeLoader.style.display = 'none';
        }, 1000);
      });
      
      // Reset form
      resetBtn.addEventListener('click', function() {
        form.reset();
        resultCard.style.display = 'none';
      });
      
      // Print report
      printBtn.addEventListener('click', function() {
        window.print();
      });
      
      // Analyze student function
      function analyzeStudent() {
        // Get form values
        const gad7Score = parseInt(document.getElementById('gadScore').value);
        const gpa = parseFloat(document.getElementById('gpa').value);
        const attendance = parseFloat(document.getElementById('attendance').value);
        const psychHistory = document.querySelector('input[name="psychHistory"]:checked').value;
        const failedCourses = parseInt(document.getElementById('failedCourses').value);
        
        // Calculate risk score according to specified logic
        let riskScore = 0;
        let factors = [];
        
        if (gad7Score > 10) {
          riskScore += 50;
          factors.push('GAD-7 score above threshold (50 points)');
        }
        
        if (gpa < 11) {
          riskScore += 20;
          factors.push('Low GPA / academic performance (20 points)');
        }
        
        if (attendance < 85) {
          riskScore += 15;
          factors.push('Low attendance rate (15 points)');
        }
        
        if (psychHistory === 'yes') {
          riskScore += 15;
          factors.push('Previous psychological history (15 points)');
        }
        
        // Add failed courses to risk calculation
        if (failedCourses > 0) {
          const failedCoursesPoints = failedCourses * 5;
          riskScore += failedCoursesPoints;
          factors.push(`Failed course count (5 points each)`);
        }
        
        // Determine risk level and set UI
        let riskLevel, color, icon, message, recommendation;
        
        if (riskScore > 50) {
          riskLevel = 'HIGH RISK';
          color = 'risk-high';
          icon = 'fa-exclamation-triangle';
          message = 'HIGH RISK DETECTED. Schedule intervention immediately.';
          recommendation = 'Immediate intervention is recommended. Schedule a one-on-one session with the student within 48 hours. Consider notifying parents/guardians and coordinating with teachers to develop an academic accommodation plan if needed.';
        } else if (riskScore >= 20) {
          riskLevel = 'MODERATE RISK';
          color = 'risk-moderate';
          icon = 'fa-exclamation-circle';
          message = 'MODERATE RISK. Monitor student.';
          recommendation = 'Regular monitoring is recommended. Schedule a check-in within the next week. Consider a follow-up GAD-7 assessment in 2-3 weeks and develop a preliminary support plan.';
        } else {
          riskLevel = 'LOW RISK';
          color = 'risk-low';
          icon = 'fa-check-circle';
          message = 'LOW RISK. Standard follow-up.';
          recommendation = 'Standard follow-up procedures are sufficient. Include student in routine wellness check-ins and preventative mental health programs. Re-evaluate if any significant academic or behavioral changes occur.';
        }
        
        // Update the result card
        resultHeader.className = `result-header ${color}`;
        resultIcon.className = `fas ${icon} risk-icon`;
        resultTitle.textContent = riskLevel;
        resultMessage.textContent = message;
        
        // Display risk factors
        riskFactors.innerHTML = '';
        if (factors.length > 0) {
          factors.forEach(factor => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = factor;
            riskFactors.appendChild(li);
          });
        }
        
        // Update recommendation text
        recommendationText.textContent = recommendation;
        
        // Show the result card
        resultCard.style.display = 'block';
      }
    });