/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#1E1E1E',
        surface: '#252526',
        nav:     '#2D2D2D',
        accent:  '#007ACC',
        border:  '#3F3F46',
      },
    },
  },
  plugins: [],
}
