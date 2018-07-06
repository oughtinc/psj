import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
GEN_DATA_DIR = os.path.join(DATA_DIR, 'generated')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
FERMI_GEN_DATA_DIR = os.path.join(GEN_DATA_DIR, 'fermi')
ANALYSIS_DATA_DIR = os.path.join(DATA_DIR, 'analysis')
FERMI_ANALYSIS_DATA_DIR = os.path.join(ANALYSIS_DATA_DIR, 'fermi')
POLITIFACT_ANALYSIS_DATA_DIR = os.path.join(ANALYSIS_DATA_DIR, 'politifact')
POLITIFACT_GEN_DATA_DIR = os.path.join(GEN_DATA_DIR, 'politifact')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
FERMI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'fermi')
POLITIFACT_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'politifact')
HUMAN_DATA_DIR = os.path.join(DATA_DIR, 'human-results')
FERMI_HUMAN_DATA_DIR = os.path.join(DATA_DIR, 'human-results', 'fermi')
POLITIFACT_HUMAN_DATA_DIR = os.path.join(DATA_DIR, 'human-results', 'politifact')
PAPERS_HUMAN_DATA_DIR = os.path.join(DATA_DIR, 'human-results', 'papers')
MODEL_DIR = os.path.join(ROOT_DIR, os.path.join('src', 'models'))
LANG_MODEL_PRETRAINED_DIR = os.path.join(MODEL_DIR, os.path.join('lang_model', 'pretrained'))
SYNTHETIC_ANALYSIS_DATA_DIR = os.path.join(ANALYSIS_DATA_DIR, 'synthetic')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
