/**
 * Security Researcher Portfolio - Main JavaScript
 * Handles cursor glow effect and smooth scrolling
 */

(function () {
  'use strict';

  // ===== CURSOR GLOW EFFECT =====
  const glow = document.getElementById('glow');
  
  if (glow) {
    // Use requestAnimationFrame for smoother animation
    let mouseX = 0;
    let mouseY = 0;
    let glowX = 0;
    let glowY = 0;
    const followSpeed = 0.15;

    function updateGlowPosition() {
      // Smooth follow effect
      glowX += (mouseX - glowX) * followSpeed;
      glowY += (mouseY - glowY) * followSpeed;
      
      glow.style.left = glowX + 'px';
      glow.style.top = glowY + 'px';
      
      requestAnimationFrame(updateGlowPosition);
    }

    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    });

    // Start the animation loop
    updateGlowPosition();
  }

  // ===== SMOOTH SCROLL FOR ANCHOR LINKS =====
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // ===== REDUCED MOTION PREFERENCE =====
  const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  
  function handleReducedMotionChange(e) {
    const glow = document.getElementById('glow');
    if (e.matches && glow) {
      // Hide glow effect when reduced motion is preferred
      glow.style.display = 'none';
    } else if (glow) {
      glow.style.display = 'block';
    }
  }

  // Initialize
  handleReducedMotionChange(mediaQuery);
  mediaQuery.addEventListener('change', handleReducedMotionChange);

  // ===== CONSOLE EASTER EGG =====
  console.log('%c🔍 Security Researcher Portfolio', 'color: #c9a84c; font-size: 16px; font-weight: bold;');
  console.log('%c"A gyertya fénye elegendő ahhoz, hogy megtaláljuk amit mások nem látnak."', 'color: #f0e8d8; font-style: italic;');
})();
