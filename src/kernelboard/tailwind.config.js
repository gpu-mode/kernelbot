/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        primary: '#FEE832',
        secondary: '#D3D93B',
        accent: '#F29999',
        neutral: '#C9C9C9',
        dark: '#202020'
      }
    }
  }
}