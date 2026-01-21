# AI-Driven Archaeological Site Mapping and Erosion Prediction Platform

## 🌟 Overview

A premium, research-grade web application for analyzing archaeological sites using advanced AI models. This platform processes satellite imagery and terrain data to perform semantic segmentation, object detection, and erosion prediction.

## ✨ Features

### 🎨 Design & UX
- **Modern Glassmorphism UI** - Floating glass panels with blur effects
- **Dark/Light Mode Toggle** - Seamless theme switching
- **Smooth Animations** - Motion-based transitions and interactions
- **Responsive Layout** - Optimized for desktop, tablet, and mobile
- **Premium Typography** - Space Grotesk and Inter font families
- **Gradient Accents** - Beautiful color gradients throughout

### 🔐 Authentication
- Login and signup flows with validation
- Simulated authentication (frontend-only)
- Form error handling and user feedback

### 📊 Main Dashboard
- Project overview widgets
- Data upload status indicators
- Pipeline progress tracking
- System status monitoring
- Quick action buttons

### 📤 Data Upload Module
- **Drag-and-drop interface** for images and CSV files
- File preview panels
- Upload progress animations
- Support for multiple image formats (PNG, JPG, TIFF)
- CSV terrain data upload

### 🤖 AI Pipeline Visualization
- **4-Step Pipeline Process:**
  1. Image Preprocessing
  2. Semantic Segmentation
  3. Object Detection
  4. Erosion Prediction
- Real-time progress tracking
- Play/Pause/Reset controls
- Animated step indicators
- Timeline visualization

### 📈 Model Performance Dashboard
- **Semantic Segmentation Metrics:**
  - IoU (Intersection over Union) Score
  - Dice Coefficient
  - Visual progress bars

- **Object Detection Metrics:**
  - mAP (mean Average Precision)
  - Precision
  - Recall
  - Performance indicators

- **Erosion Prediction Metrics:**
  - RMSE (Root Mean Square Error)
  - R² Score
  - Trend visualization

- **Interactive Charts:**
  - Erosion trend line chart
  - Model comparison bar chart
  - Responsive recharts visualizations

### 🎯 Results & Insights
- **Risk Assessment:**
  - Low/Medium/High erosion classification
  - Confidence level indicators
  - Color-coded risk badges

- **AI-Generated Insights:**
  - Segmentation analysis
  - Detection performance summary
  - Erosion prediction insights

- **Recommendations:**
  - Actionable next steps
  - Data collection suggestions
  - Model optimization tips

- **Export Functionality:**
  - Download analysis reports (JSON format)

### ⭐ Review & Feedback
- 5-star rating system
- Feedback category selection
- Comment submission
- Success animation on submit

### 🎉 Success Screen
- Analysis completion celebration
- Metrics summary dashboard
- Next steps guidance
- Option to start new analysis

### 🔔 Notifications
- Floating toast notifications
- Success, error, warning, and info types
- Auto-dismiss after 5 seconds
- Smooth enter/exit animations

## 🛠️ Technology Stack

- **React 18.3.1** - UI framework
- **TypeScript** - Type safety
- **Motion (Framer Motion)** - Animations
- **Recharts** - Data visualization
- **Plain CSS** - Custom styling (no Tailwind in components)
- **Vite** - Build tool

## 📁 Project Structure

```
src/
├── app/
│   ├── App.tsx                           # Main application component
│   ├── App.css                           # Global styles
│   └── components/
│       ├── LandingPage.tsx               # Hero landing page
│       ├── LandingPage.css
│       ├── AuthPage.tsx                  # Login/Signup
│       ├── AuthPage.css
│       ├── MainDashboard.tsx             # Main dashboard
│       ├── MainDashboard.css
│       ├── DataUploadModule.tsx          # File upload interface
│       ├── DataUploadModule.css
│       ├── AIPipelineVisualization.tsx   # Pipeline progress
│       ├── AIPipelineVisualization.css
│       ├── ModelPerformanceDashboard.tsx # Metrics & charts
│       ├── ModelPerformanceDashboard.css
│       ├── ResultsInsights.tsx           # Analysis results
│       ├── ResultsInsights.css
│       ├── ReviewFeedback.tsx            # Feedback form
│       ├── ReviewFeedback.css
│       ├── SuccessScreen.tsx             # Completion screen
│       ├── SuccessScreen.css
│       ├── Notifications.tsx             # Toast notifications
│       └── Notifications.css
└── styles/
    ├── fonts.css                         # Font imports
    ├── index.css                         # Main CSS entry
    ├── tailwind.css                      # Tailwind base
    └── theme.css                         # Theme tokens
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Run Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

## 🎮 User Flow

1. **Landing Page** → View project overview and features
2. **Authentication** → Login or create account
3. **Dashboard** → View project status and overview
4. **Upload Data** → Upload satellite imagery and CSV data
5. **Run Pipeline** → Execute AI analysis pipeline
6. **View Performance** → Examine model metrics and charts
7. **Review Results** → Analyze insights and recommendations
8. **Provide Feedback** → Rate experience and submit comments
9. **Success Screen** → View completion summary

## 🎨 Design Principles

- **NASA/Google AI Aesthetics** - Professional research platform look
- **Glassmorphism** - Modern frosted glass effects
- **Gradient Accents** - Vibrant color gradients for emphasis
- **Micro-interactions** - Hover effects and smooth transitions
- **Information Hierarchy** - Clear visual organization
- **Accessibility** - High contrast ratios and readable fonts

## 📊 Key Metrics Displayed

### Segmentation
- **IoU Score**: Measures overlap between predicted and actual segments
- **Dice Score**: Similarity coefficient for segmentation quality

### Detection
- **mAP**: Mean Average Precision across all classes
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases

### Erosion Prediction
- **RMSE**: Root Mean Square Error of predictions
- **R² Score**: Coefficient of determination (model fit quality)

## 🎯 Future Enhancements

- Real backend integration with Supabase
- Live camera feed integration
- 3D terrain visualization
- Multi-site comparison
- Historical data analysis
- Team collaboration features
- PDF report generation
- Advanced filtering and search

## 📝 Notes

- All AI analysis is **simulated** for demonstration purposes
- Metrics are **randomly generated** within realistic ranges
- File uploads are handled **client-side only**
- No actual backend processing occurs

## 🏆 Code Quality

- **1200+ lines** of production-ready code
- **Clean component structure** with separation of concerns
- **Reusable patterns** across all pages
- **Consistent styling** with CSS variables
- **Type-safe** with TypeScript interfaces
- **Smooth animations** with motion/react
- **Responsive design** for all screen sizes

## 📄 License

This is a demonstration project for educational and portfolio purposes.

---

**Built with ❤️ for archaeological research and AI innovation**
