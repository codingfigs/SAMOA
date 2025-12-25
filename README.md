# SAMOA - Statistical Application for Morphological & Ontological Analysis

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-Commercial-red.svg)

**SAMOA** is a comprehensive desktop application for advanced text analysis, combining morphological analysis, ontological analysis, and statistical methods to extract meaningful insights from textual data.

---

## 🎯 Key Features

### 📊 Data Import & Management
- **Supported Formats**: CSV, Excel (.xlsx, .xls)
- **CRUD Operations**: Full Create, Read, Update, Delete functionality
- **Data Preview**: Interactive table with 100+ rows display
- **Column Detection**: Automatic identification of text columns (Abstract/Abstracts)
- **Real-time Editing**: Modify data directly in the application

### 🔤 Morphological Analysis
- **Morphological Features**: Extract linguistic features (nouns, verbs, affixes)
- **Word Families**: Identify related word forms and lemmatization
- **Domain Patterns**: Discover field-specific morphological patterns
- **POS Tagging**: Part-of-speech analysis using NLTK
- **Lemmatization**: WordNet-based word normalization

### 🧠 Ontological Analysis
- **Entity Extraction**: Identify key domain-specific terms
- **Concept Hierarchies**: Build IS-A relationships
- **Attribute Ontology**: Extract adjective-noun pairs
- **Relationship Ontology**: Discover semantic relationships
- **Process Ontology**: Identify verb-noun process patterns
- **Problem-Solution Analysis**: Map challenges to solutions
- **Evaluation Ontology**: Extract assessment terms

### 📈 Statistical Analysis
- **DTM Generation**: Document-Term Matrix with customizable parameters
- **TF-IDF Generation**: Term Frequency-Inverse Document Frequency
- **Descriptive Statistics**: Mean, median, std dev, quartiles
- **Statistical Tests**: T-Test, Chi-Square, ANOVA, Correlation
- **Regression Models**: Linear, Logistic, Random Forest, Decision Tree, SVM, Neural Networks
- **Clustering**: K-Means with silhouette analysis
- **Dimensionality Reduction**: PCA, EFA, Correspondence Analysis
- **CFA/SEM**: Confirmatory Factor Analysis with CB-SEM and PLS-SEM

### 📤 Export & Reporting
- **Multiple Formats**: CSV, Excel, Word (.docx), PDF
- **Comprehensive Reports**: Tables, plots, and statistical results
- **Batch Export**: Export all analyses at once
- **Custom Reports**: Select specific sections to include
- **High-Quality Plots**: Publication-ready visualizations

### 🎨 Visualization
- **Interactive Plots**: Bar charts, heatmaps, scatter plots
- **Word Clouds**: Visual term frequency representation
- **Treemaps**: Hierarchical data visualization
- **Network Graphs**: Relationship visualization
- **Statistical Charts**: Box plots, Q-Q plots, residual plots

---

## 🚀 Getting Started

### Installation

#### Option 1: Windows Installer (Recommended)
1. Download `SAMOA_Setup_v1.0.0.exe`
2. Run the installer
3. Follow installation wizard
4. Launch from Start Menu

#### Option 2: Portable Version
1. Download and extract `SAMOA_Portable.zip`
2. Run `SAMOA.exe`
3. No installation required

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 500 MB free space
- **Display**: 1366x768 minimum resolution

---

## 📖 User Guide

### 1. Import Data

**Step 1**: Click **📁 Import Data**
- Select CSV or Excel file
- Supported encodings: UTF-8, Latin-1
- Automatic column detection

**Step 2**: Preview Data
- View first 100 rows
- Edit cells directly (right-click for options)
- Add/delete rows and columns

**Supported Data Structure**:
```
| Title | Abstract | Author | Year | Keywords |
|-------|----------|--------|------|----------|
| ...   | ...      | ...    | ...  | ...      |
```

**Note**: Application analyzes text from **Abstract** or **Abstracts** columns only.

### 2. Configure Preprocessing

**Click**: ⚙️ Preprocessing

**Options**:
- ✅ Remove stop words (the, a, this, etc.)
- ✅ Minimum word length (default: 3)
- ✅ Convert to lowercase
- ✅ Custom stop words (comma-separated)

**Extended Stop Words**: Includes 200+ academic and generic terms for cleaner analysis.

### 3. Run Analysis

**Click**: 🔬 Run Analysis

**Processes**:
1. Text tokenization and cleaning
2. POS tagging and lemmatization
3. Domain term extraction
4. Morphological feature analysis
5. Ontological relationship mapping

**Output**: 11 analysis tabs with tables and visualizations

### 4. Statistical Analysis

**Click**: 📊 Statistical Analysis

#### Generate Term Vectors

**DTM (Document-Term Matrix)**:
- Click **Generate DTM**
- Configure options: n-grams, max features, min/max DF
- Output: Document × Term frequency matrix

**TF-IDF**:
- Click **Generate TF-IDF**
- Same configuration options
- Output: Weighted term importance matrix

#### Descriptive Statistics
- Summary statistics for all variables
- Distribution plots
- Export to CSV/Excel/Word/PDF

#### Statistical Tests
- **T-Test**: Compare two groups
- **Chi-Square**: Test independence
- **ANOVA**: Compare multiple groups
- **Correlation**: Pearson/Spearman correlation matrix

#### Predictive Models

**Regression**:
- Linear Regression (with coefficients)
- Random Forest (with feature importance)
- Decision Tree (with tree visualization)
- SVM (with support vectors)
- Neural Network (with architecture diagram)

**Classification**:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- SVM Classifier
- Neural Network Classifier

**Clustering**:
- K-Means with elbow plot
- Silhouette analysis
- Cluster visualization

#### Multivariate Analysis
- **Correlation Matrix**: Heatmap visualization
- **PCA**: Principal Component Analysis with scree plot
- **EFA**: Exploratory Factor Analysis with loadings
- **CFA/SEM**: Confirmatory Factor Analysis (CB-SEM, PLS-SEM)
- **Correspondence Analysis**: Chi-square based dimension reduction

### 5. Export Results

#### Individual Tab Export
- Right-click on any table → Export to CSV/Excel/Word/PDF
- Plots automatically included in Word/PDF exports

#### Batch Export
**Click**: 💾 Export All

**Options**:
- Export All to Excel (single file, multiple sheets)
- Export All to CSV (multiple files)
- Export All to Word (comprehensive report)
- Export All to PDF (publication-ready)

**Report Contents**:
- ✅ Data preview
- ✅ Morphological features
- ✅ Word families
- ✅ Domain patterns
- ✅ All ontological analyses
- ✅ Plots and visualizations

---

## 🎨 Analysis Tabs Overview

### Tab 1: Data Preview (CRUD)
- View and edit imported data
- Add/delete rows and columns
- Export to CSV/Excel

### Tab 2: Morphological Features
- Token frequency with linguistic features
- POS tags (Noun, Verb, Adjective)
- Morphological affixes (prefixes, suffixes)

### Tab 3: Word Families
- Lemma-based word grouping
- Family examples and frequency
- POS distribution

### Tab 4: Domain Patterns
- Field-specific morphological patterns
- Suffix analysis (-ology, -ization, -ism)
- Pattern frequency

### Tab 5: Entity Extraction
- Top domain-specific terms
- Word cloud visualization
- Frequency distribution

### Tab 6: Concept Hierarchies
- IS-A relationships
- Parent-child concepts
- Network graph visualization

### Tab 7: Attribute Ontology
- Adjective-noun pairs
- Attribute frequency
- Treemap visualization

### Tab 8: Relationship Ontology
- Semantic relationships
- Verb-based relations
- Frequency analysis

### Tab 9: Process Ontology
- Verb-noun process patterns
- Action-object relationships
- Process frequency

### Tab 10: Problem-Solution
- Problem terms identification
- Solution terms mapping
- Ratio analysis

### Tab 11: Evaluation Ontology
- Assessment terms (accuracy, performance)
- Quality measures
- Evaluation frequency

---

## 🔧 Advanced Features

### Python Console
**Access**: 🐍 Python Console button

**Available Objects**:
- `df` - Your imported DataFrame
- `pd` - pandas library
- `np` - numpy library
- `plt` - matplotlib.pyplot
- `analyzer` - Analysis engine

**Example Usage**:
```python
# View data
df.head()
df.describe()

# Custom analysis
df['Abstract'].str.len().mean()

# Custom plots
plt.figure()
df['Year'].value_counts().plot(kind='bar')
plt.show()
```

### Variable Selection
**Access**: 📋 Select Variables (in Statistical Analysis)

**Features**:
- Search and filter from ALL corpus terms
- Multi-select with Ctrl/Shift
- Transfer selected variables to analysis
- Replace or append to existing data

### Recode/Transform
**Right-click** on any column → Recode/Transform Column

**Options**:
- Binning (equal width/frequency)
- Z-score normalization
- Log transformation
- Custom mapping

---

## 📊 Use Cases

### Academic Research
- Literature review analysis
- Systematic review synthesis
- Bibliometric analysis
- Concept mapping
- Trend identification

### Business Intelligence
- Customer feedback analysis
- Product review mining
- Market research
- Competitive analysis
- Brand perception

### Content Analysis
- Social media analysis
- News article analysis
- Policy document analysis
- Survey response analysis
- Qualitative data coding

### NLP Research
- Corpus linguistics
- Semantic analysis
- Discourse analysis
- Text classification
- Feature engineering

---

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **GUI**: PyQt5
- **NLP**: NLTK
- **Data**: pandas, numpy
- **ML**: scikit-learn
- **Stats**: scipy, semopy
- **Viz**: matplotlib, seaborn, wordcloud, networkx
- **Export**: python-docx, openpyxl, reportlab

---

## 📝 Data Format Guidelines

### Recommended CSV/Excel Structure

```csv
Title,Abstract,Author,Year,Keywords,Journal
"Paper Title","Abstract text here...","Author Name",2024,"keyword1, keyword2","Journal Name"
```

### Best Practices
- ✅ Use UTF-8 encoding for CSV files
- ✅ Include column headers in first row
- ✅ Name text column as "Abstract" or "Abstracts"
- ✅ Ensure abstracts have sufficient text (50+ words)
- ✅ Remove special characters if causing import issues
- ❌ Avoid merged cells in Excel
- ❌ Avoid formulas in data cells

---

## 🎓 Workflow Example

### Research Paper Analysis

**Step 1**: Import Data
- Load CSV with 100 research papers
- Columns: Title, Abstract, Year, Keywords

**Step 2**: Preprocess
- Enable stop word removal
- Set minimum word length: 3
- Add custom stop words: "study, research, paper"

**Step 3**: Morphological Analysis
- Identify key morphological features
- Analyze word families
- Discover domain patterns

**Step 4**: Ontological Analysis
- Extract key entities (concepts)
- Build concept hierarchies
- Map relationships

**Step 5**: Statistical Analysis
- Generate TF-IDF matrix
- Run PCA for dimensionality reduction
- Cluster papers using K-Means
- Identify research themes

**Step 6**: Export
- Export comprehensive Word report
- Include all tables and visualizations
- Share with research team

---

## 🤝 Support & Contact

**Developer**: Dr. M. Kamakshaiah  
**Company**: AMCHIK SOLUTIONS, India  
**Email**: contact@codingfigs.com  
**Copyright**: © 2024 AMCHIK SOLUTIONS

---

## 📄 Citation

If you use SAMOA in your research, please cite:

```
SAMOA - Statistical Application for Morphological & Ontological Analysis
Version 1.0.0 (2024)
Developer: Dr. M. Kamakshaiah
AMCHIK SOLUTIONS, India
```

---

## 🔄 Version History

### Version 1.0.0 (2024)
- Initial release
- Morphological analysis engine
- Ontological analysis engine
- Statistical analysis suite
- CFA/SEM integration
- Multi-format export
- Python console
- Comprehensive documentation

---

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Cloud integration
- [ ] Real-time collaboration
- [ ] Advanced NLP models (BERT, GPT)
- [ ] Custom model training
- [ ] API access
- [ ] Web version

---

**Made with ❤️ for researchers and analysts worldwide**
