LIGHT_STYLESHEET = r"""
QWidget { color: #172033; font-family: "Microsoft YaHei UI"; font-size: 13px; }
QMainWindow, QWidget#root, QWidget#devicePage, QWidget#dashboard,
QWidget#experimentPage, QWidget#experimentContent, QWidget#dataCheckPage,
QWidget#dataCheckContent, QWidget#datasetUploadPage,
QWidget#datasetUploadContent { background: #eef3f8; }
QWidget#experimentLeftColumn { background: transparent; }
QTabWidget::pane { background: #eef3f8; border: none; border-top: 1px solid #d8e1ec; }
QTabBar::tab { min-width: 108px; min-height: 36px; padding: 0 18px; margin-right: 4px;
    background: #dfe8f2; color: #344054; border: 1px solid #cbd7e5; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; }
QTabBar::tab:selected { background: #ffffff; color: #175cd3; }
QTabBar::tab:hover { background: #f8fbff; color: #175cd3; }
QFrame#header { background: #ffffff; border-bottom: 1px solid #dfe6ef; }
QLabel#appTitle { color: #101828; font-size: 22px; font-weight: 700; }
QLabel#appSubtitle { color: #667085; font-size: 12px; }
QLabel#sectionTitle { color: #344054; font-size: 12px; font-weight: 700; }
QLabel#metricValue { color: #101828; font-size: 17px; font-weight: 700; }
QLabel#muted { color: #667085; }
QLabel#pageTitle { color: #101828; font-size: 20px; font-weight: 700; }
QLabel#pageSubtitle, QLabel#cardDescription { color: #667085; font-size: 12px; }
QLabel#cardTitle { color: #344054; font-size: 14px; font-weight: 700; }
QLabel#fieldLabel { color: #475467; font-size: 13px; font-weight: 600; min-width: 92px; }
QLabel#stateIdle, QLabel#stateRunning { padding: 7px 13px; border-radius: 14px; font-weight: 700; }
QLabel#stateIdle { color: #475467; background: #e4e7ec; }
QLabel#stateRunning { color: #067647; background: #d1fadf; }
QLabel#signalResult { color: #667085; padding: 7px 10px; background: #f8fafc;
    border: 1px solid #e2e8f0; border-radius: 8px; }
QFrame#qualityWaiting, QFrame#qualityGood, QFrame#qualityWarning, QFrame#qualityBad {
    border-radius: 8px; }
QFrame#qualityWaiting { background: #f8fafc; border: 1px solid #e2e8f0; }
QFrame#qualityGood { background: #ecfdf3; border: 1px solid #86efac; }
QFrame#qualityWarning { background: #fffaeb; border: 1px solid #fedf89; }
QFrame#qualityBad { background: #fef3f2; border: 1px solid #fecdca; }
QLabel#qualityChannel { color: #475467; font-size: 11px; font-weight: 700; border: none; }
QLabel#qualityStatus { color: #101828; font-size: 14px; font-weight: 700; border: none; }
QLabel#qualityMetrics { color: #667085; font-size: 10px; border: none; }
QLabel#statusTitle { color: #475467; font-size: 12px; font-weight: 700; }
QLabel#statusTitleAccent { color: #175cd3; font-size: 12px; font-weight: 700; }
QLabel#statusText { color: #344054; font-size: 14px; line-height: 1.5; }
QLabel#progressText { color: #101828; font-size: 16px; font-weight: 700; line-height: 1.5; }
QLabel#dataStatusIdle, QLabel#dataStatusRunning, QLabel#dataStatusGood, QLabel#dataStatusBad,
QLabel#validationIdle, QLabel#validationGood, QLabel#validationBad {
    padding: 9px 12px; border-radius: 8px; font-weight: 600; }
QLabel#dataStatusIdle, QLabel#validationIdle { color: #475467; background: #f8fafc; border: 1px solid #e2e8f0; }
QLabel#dataStatusRunning { color: #175cd3; background: #eff8ff; border: 1px solid #b2ddff; }
QLabel#dataStatusGood, QLabel#validationGood { color: #067647; background: #ecfdf3; border: 1px solid #abefc6; }
QLabel#dataStatusBad, QLabel#validationBad { color: #b42318; background: #fef3f2; border: 1px solid #fecdca; }
QFrame#qcMetricNeutral, QFrame#qcMetricGood, QFrame#qcMetricBad, QFrame#qcMetricAccent { border-radius: 9px; }
QFrame#qcMetricNeutral { background: #f8fafc; border: 1px solid #e2e8f0; }
QFrame#qcMetricGood { background: #ecfdf3; border: 1px solid #abefc6; }
QFrame#qcMetricBad { background: #fef3f2; border: 1px solid #fecdca; }
QFrame#qcMetricAccent { background: #eff8ff; border: 1px solid #b2ddff; }
QLabel#qcMetricLabel { color: #667085; font-size: 11px; font-weight: 600; border: none; }
QLabel#qcMetricValue { color: #101828; font-size: 20px; font-weight: 700; border: none; }
QFrame#card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; }
QFrame#statusPanel { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px; }
QFrame#statusPanelAccent { background: #eff8ff; border: 1px solid #b2ddff; border-radius: 9px; }
QFrame#divider { color: #e2e8f0; background: #e2e8f0; max-height: 1px; border: none; }
QWidget#embeddedPrompt { background: #f7f9fc; border: 1px solid #d8e1ec; border-radius: 12px; }
QWidget#navigationCanvas { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }
QLabel#promptState { color: #101828; font-size: 22px; font-weight: 700; }
QLabel#promptProgress { color: #344054; font-size: 17px; font-weight: 700;
    padding: 6px 11px; background: #eef2f7; border-radius: 8px; }
QLabel#taskStatusIdle, QLabel#taskStatusRunning, QLabel#taskStatusCollecting { padding: 6px 10px; border-radius: 14px; font-weight: 700; }
QLabel#taskStatusIdle { color: #475467; background: #e4e7ec; }
QLabel#taskStatusRunning { color: #067647; background: #d1fadf; }
QLabel#taskStatusCollecting { color: #b42318; background: #fee4e2; }
QLabel#taskCollectionPrompt { color: #1d4ed8; background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 9px; padding: 10px 14px; font-size: 15px; font-weight: 700; }
QLabel#taskCollectionPrompt[fingerTone="thumb"] { color: #2563eb; }
QLabel#taskCollectionPrompt[fingerTone="index"] { color: #e11d48; }
QLabel#taskCollectionPrompt[fingerTone="middle"] { color: #7c3aed; }
QLabel#taskCollectionPrompt[fingerTone="neutral"] { color: #475467; }
QFrame#signalPlotRow { background: #ffffff; border: none; }
QGroupBox { background: white; border: 1px solid #d8e1ec; border-radius: 10px; margin-top: 12px;
    padding: 12px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton { padding: 0 14px; border-radius: 8px; border: 1px solid #d0d5dd;
    background: #ffffff; color: #344054; font-weight: 600; }
QPushButton:hover { background: #f8fafc; border-color: #98a2b3; }
QPushButton:pressed { background: #eef2f7; }
QPushButton:disabled { color: #98a2b3; background: #f2f4f7; border-color: #eaecf0; }
QPushButton:checked { background: #2563eb; color: #ffffff; border-color: #2563eb; }
QPushButton:checked:hover { background: #1d4ed8; }
QPushButton#primary { background: #2563eb; color: white; border-color: #2563eb; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { color: #98a2b3; background: #f2f4f7; border-color: #eaecf0; }
QPushButton#danger { background: #fff5f5; color: #c81e1e; border-color: #fecaca; }
QPushButton#danger:disabled { color: #98a2b3; background: #f2f4f7; border-color: #eaecf0; }
QPushButton#record { background: #e11d48; color: white; border-color: #e11d48; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit { padding: 0 10px;
    background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px;
    selection-background-color: #bfdbfe; }
QTextEdit, QPlainTextEdit { padding: 6px; }
QProgressBar { background: #e4e7ec; border: none; border-radius: 4px; }
QProgressBar::chunk { background: #2563eb; border-radius: 4px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 1px solid #2563eb; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #98a2b3; background: #f2f4f7; border-color: #eaecf0; }
QComboBox::drop-down { border: none; width: 24px; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; background: #ffffff; border: 1px solid #98a2b3; border-radius: 4px; }
QCheckBox::indicator:hover { border-color: #2563eb; }
QCheckBox::indicator:checked { background: #2563eb; border-color: #2563eb; }
QScrollArea { border: none; background: #eef3f8; }
QScrollArea > QWidget > QWidget, QWidget#sidebarContent { background: #eef3f8; }
QScrollBar:vertical { width: 12px; margin: 0; background: #eef3f8; }
QScrollBar:horizontal { height: 12px; margin: 0; background: #eef3f8; }
QScrollBar::handle { min-width: 28px; min-height: 28px; background: #b8c5d6; border-radius: 6px; }
QScrollBar::handle:hover { background: #98a8bc; }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent; border: none; }
QStatusBar { background: #ffffff; color: #667085; border-top: 1px solid #e2e8f0; }
QToolTip { background: #101828; color: #ffffff; border: none; padding: 6px; }
"""
