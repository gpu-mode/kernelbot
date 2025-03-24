/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./kernelboard/templates/**/*.html"],
  safelist: [ 'code-block-fade' ],
  theme: {
    extend: {
      colors: {
        primary: '#FEE832',   // Yellow
        secondary: '#D3D93B', // Green
        accent: '#F29999',    // Coral
        neutral: '#C9C9C9',   // Gray
        dark: '#202020',      // Dark Gray
        discord: '#5865F2',   // Discord's brand color, blurple
      }
    }
  }
}