import sys
import pandas as pd
import numpy as np
import traceback
import re
from scipy.interpolate import griddata

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox, QSlider,
    QSizePolicy, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib.colors import LogNorm, Normalize

# --- Machine Learning Imports ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# --- Constants ---
FIXED_COLOR_MIN_DEFAULT = 1.0e-15 # Default for fixed log
FIXED_COLOR_MAX_DEFAULT = 1.0e-02 # Default for fixed log
LOG_FLOOR = 1e-15
CATEGORICAL_THRESHOLD = 15 # Max unique values for a column to be treated as categorical
# --- Isotope Regex Pattern ---
ISOTOPE_PATTERN = re.compile(r'^[A-Za-z]{1,2}-\d{1,3}m?(?:(?: \(.+?\))|(?:\.\d+))?$')
# --- Simplified base name pattern for grouping ---
ISOTOPE_BASE_PATTERN = re.compile(r'^([A-Za-z]{1,2}-\d{1,3}m?(?: \(.+?\))?)') # Captures base name


# --- ChartWindow Class ---
class ChartWindow(QMainWindow):
    def __init__(self, selected_isotope_prefix, plot_type,
                 var_axis1, var_axis2, var_axis3_or_mode, # var_axis2/3 might be None
                 var_color, # For Bubble/Bar
                 slider_var, # Slider filter variable
                 initial_slider_value,
                 importance_df, isotope_df,
                 value_col, # Target numeric column FOR PLOTTING (user choice)
                 categorical_vars_list, continuous_vars_list, # Identified variable types
                 scale_type, range_type, fixed_min, fixed_max # Scale options
                 ):
        super().__init__()

        # --- Store parameters ---
        self.selected_isotope_prefix = selected_isotope_prefix
        self.plot_type = plot_type
        self.var_x = var_axis1
        self.var_y = var_axis2
        self.var_z_or_mode = var_axis3_or_mode
        self.var_color = var_color
        self.slider_control_var_name = slider_var
        self.importance_df = importance_df
        self.isotope_df = isotope_df
        self.value_col = value_col # This is now the user-selected target for plotting
        self.slider_categories = []
        self.categorical_variables = categorical_vars_list
        self.continuous_variables = continuous_vars_list
        self.scale_type = scale_type
        self.range_type = range_type
        self.fixed_min = fixed_min
        self.fixed_max = fixed_max
        print(f"ChartWindow received plotting target: '{self.value_col}'")
        print(f"ChartWindow received scale: {self.scale_type}, range: {self.range_type}, fixed_min: {self.fixed_min}, fixed_max: {self.fixed_max}")

        # --- Determine plot dimensionality ---
        self.plot_dimension = '2D'
        if self.plot_type == 'heatmap_scatter':
            if self.var_y is not None and self.var_z_or_mode is not None: self.plot_dimension = '3D'
        elif self.plot_type in ['bubble', 'bar3d', 'surface']: self.plot_dimension = '3D'

        # --- Define Color Maps ---
        self.heatmap_cmap = 'hot'
        self.bubble_cmap = 'tab10'
        self.bar_cmap = 'viridis'
        self.surface_cmap = 'viridis'


        self.setWindowTitle(f"{selected_isotope_prefix} - {self._get_plot_title_prefix()} ({self.plot_dimension})")
        self.setGeometry(100, 100, 1500, 850)

        # --- UI Setup ---
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(); central_widget.setLayout(main_layout)
        info_label = QLabel(self._build_info_text()) # Uses self.value_col
        main_layout.addWidget(info_label)
        charts_layout = QHBoxLayout()
        self.vi_fig = Figure(figsize=(4, 5), dpi=100); self.vi_canvas = FigureCanvas(self.vi_fig); self.vi_ax = self.vi_fig.add_subplot(111)
        charts_layout.addWidget(self.vi_canvas, 1)
        self.dynamic_plot_fig = Figure(figsize=(7, 6), dpi=100); self.dynamic_plot_canvas = FigureCanvas(self.dynamic_plot_fig); self.dynamic_plot_ax = None
        charts_layout.addWidget(self.dynamic_plot_canvas, 3)
        main_layout.addLayout(charts_layout, stretch=1)
        slider_layout_container = QVBoxLayout()
        dynamic_slider_layout = QHBoxLayout()
        slider_label_text = f"Filter by '{self.slider_control_var_name}' Category:" if self.slider_control_var_name else "Slider (No Variable Assigned):"
        dynamic_slider_layout.addWidget(QLabel(slider_label_text))
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setMinimum(0); self.slider.setMaximum(0); self.slider.setValue(initial_slider_value); self.slider.setMinimumWidth(400); self.slider.setMinimumHeight(30); self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed); self.slider.setStyleSheet("""QSlider::handle:horizontal { width: 25px; height: 25px; margin: -12px 0; background: #2A9D8F; border-radius: 12px; } QSlider::groove:horizontal { height: 8px; background: #264653; border-radius: 4px; }"""); self.slider.setEnabled(bool(self.slider_control_var_name))
        dynamic_slider_layout.addWidget(self.slider, stretch=1)
        slider_layout_container.addLayout(dynamic_slider_layout)
        self.slider_category_label = QLabel("Selected Category: All"); self.slider_category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout_container.addWidget(self.slider_category_label)
        main_layout.addLayout(slider_layout_container)
        vi_slider_layout = QHBoxLayout(); vi_slider_layout.addWidget(QLabel("Unused Slider:")); self.slider_vi = QSlider(Qt.Orientation.Horizontal); self.slider_vi.setMinimum(0); self.slider_vi.setMaximum(100); self.slider_vi.setValue(50); self.slider_vi.setMinimumWidth(400); self.slider_vi.setMinimumHeight(30); self.slider_vi.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed); self.slider_vi.setStyleSheet("""QSlider::handle:horizontal { width: 25px; height: 25px; margin: -12px 0; background: #E9C46A; border-radius: 12px; } QSlider::groove:horizontal { height: 8px; background: #264653; border-radius: 4px; }"""); self.slider_vi.setEnabled(False); vi_slider_layout.addWidget(self.slider_vi, stretch=1)
        main_layout.addLayout(vi_slider_layout)

        # --- Connect Signals ---
        self.slider.valueChanged.connect(self.plot_dynamic_chart)

        # --- Initial Plot Generation ---
        self._update_slider_range_and_label(initial_slider_value)
        self.plot_variable_importance()
        self.plot_dynamic_chart() # Plots using the selected value_col

    def _get_plot_title_prefix(self):
        """Helper to get a clean title part for the window."""
        if self.plot_type == 'heatmap_scatter': return "Heatmap"
        elif self.plot_type == 'bubble': return "3D Bubble Chart"
        elif self.plot_type == 'bar3d': return "3D Bar Chart"
        elif self.plot_type == 'surface': return "Surface Plot"
        else: return "Chart"

    def _get_value_col_usage(self):
        """Helper to describe how the numeric value column is used."""
        usage = "N/A"
        if self.plot_type == 'bubble': usage = 'Size'
        elif self.plot_type == 'bar3d': usage = 'Bar Height (Z, Fixed Log Scale)' # Height is always fixed log
        elif self.plot_type == 'surface': usage = f'Surface Height (Z, {self.scale_type}, {self.range_type})'
        elif self.plot_type == 'heatmap_scatter':
            if self.plot_dimension == '2D': usage = f'Y-axis & Color ({self.scale_type}, {self.range_type})'
            else: usage = f'Color ({self.scale_type}, {self.range_type})' # 3D Heatmap color
        return usage

    def _build_info_text(self):
        """Builds the text for the top info label."""
        plot_name = self._get_plot_title_prefix()
        text = f"Isotope Context: {self.selected_isotope_prefix}. Plot: {plot_name} ({self.plot_dimension}). "

        # Info reflects the PLOTTING target column
        text += f"Value Col (Plotting): '{self.value_col}' -> {self._get_value_col_usage()}. "
        if self.plot_type == 'bubble': text += f"Color: '{self.var_color}'. "
        if self.plot_type == 'bar3d': text += f"Color: '{self.var_color}'. "
        if self.slider_control_var_name: text += f"Slider: '{self.slider_control_var_name}'"
        return text

    def _encode_categorical(self, df, var_name, sort_by_value=False, value_col_for_sort=None):
        """Encodes a categorical variable"""
        if var_name not in df.columns:
            print(f"Warning: Column '{var_name}' not found for encoding. Returning empty.")
            return pd.Series(dtype=int), [], {}

        cat_col_str = df[var_name].astype(str)
        unique_cats = cat_col_str.unique()
        ordered_categories = []

        if sort_by_value and value_col_for_sort and value_col_for_sort in df.columns:
            numeric_vals = pd.to_numeric(df[value_col_for_sort], errors='coerce')
            if not numeric_vals.isnull().all():
                try: # Add try-except for robustness if groupby fails
                    category_means = df.loc[numeric_vals.notna() & cat_col_str.notna()].groupby(cat_col_str)[value_col_for_sort].mean()
                    ordered_categories = category_means.sort_values(ascending=False).index.tolist()
                    missing = [cat for cat in unique_cats if cat not in ordered_categories]
                    ordered_categories.extend(sorted(missing))
                except Exception as group_err:
                     print(f"Warning: Groupby for sorting '{var_name}' failed: {group_err}. Using alphabetical.")
                     ordered_categories = [] # Fallback to alphabetical

        if not ordered_categories:
            ordered_categories = sorted(unique_cats)

        if not ordered_categories:
            print(f"Warning: No categories found or determined for '{var_name}' after processing.")
            mapping = {}
        else:
            mapping = {category: i for i, category in enumerate(ordered_categories)}

        codes = cat_col_str.map(mapping).fillna(-1).astype(int)
        return codes, ordered_categories, mapping

    def _setup_categorical_axis(self, axis_letter, var_name, categories, ax):
        """Sets ticks and labels for a categorical axis"""
        ticks = list(range(len(categories)))
        label = f"{var_name} (Categorical)"
        n_cats = len(categories)
        show_step = 1
        if n_cats > 20: show_step = max(1, n_cats // 15)

        labels_to_show = [categories[i] if i % show_step == 0 else '' for i in ticks]

        if axis_letter == 'x':
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels_to_show, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel(label)
        elif axis_letter == 'y':
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels_to_show, rotation=0, ha="right", fontsize=8) # Adjusted rotation
            ax.set_ylabel(label)
        elif axis_letter == 'z':
            ax.set_zticks(ticks)
            ax.set_zticklabels(labels_to_show, rotation=-15, ha="left", fontsize=8)
            ax.set_zlabel(label)

    def _update_slider_range_and_label(self, initial_slider_index=0):
        """Sets slider max based on number of categories and updates initial label."""
        self.slider_categories = []
        if self.slider_control_var_name and self.slider_control_var_name in self.isotope_df.columns:
            if self.slider_control_var_name in self.categorical_variables:
                try:
                    unique_cats = sorted(self.isotope_df[self.slider_control_var_name].astype(str).unique())
                    self.slider_categories = unique_cats
                    num_categories = len(unique_cats)
                    if num_categories > 0:
                        self.slider.setMaximum(num_categories - 1)
                        safe_initial_index = max(0, min(initial_slider_index, num_categories - 1))
                        self.slider.setValue(safe_initial_index)
                        current_index = self.slider.value()
                        if 0 <= current_index < len(self.slider_categories):
                             self.slider_category_label.setText(f"Selected: {self.slider_categories[current_index]}")
                        else: self.slider_category_label.setText(f"Selected: Invalid Index {current_index}")
                        self.slider.setEnabled(True)
                        print(f"Slider updated: {num_categories} categories for '{self.slider_control_var_name}'. Initial: '{self.slider_categories[safe_initial_index]}'")
                    else:
                        self.slider.setMaximum(0); self.slider.setValue(0); self.slider.setEnabled(False)
                        self.slider_category_label.setText("Selected: N/A (No categories found)")
                        print(f"Slider disabled: No categories found for '{self.slider_control_var_name}'.")
                except Exception as e:
                    print(f"Error updating slider for '{self.slider_control_var_name}': {e}"); traceback.print_exc()
                    self.slider.setMaximum(0); self.slider.setValue(0); self.slider.setEnabled(False)
                    self.slider_category_label.setText("Selected: Error updating slider")
            else:
                self.slider.setMaximum(0); self.slider.setValue(0); self.slider.setEnabled(False)
                self.slider_category_label.setText(f"Selected: N/A ('{self.slider_control_var_name}' is Continuous)")
                print(f"Slider disabled: Control variable '{self.slider_control_var_name}' is continuous.")
        else:
            self.slider.setMaximum(0); self.slider.setValue(0); self.slider.setEnabled(False)
            self.slider_category_label.setText("Selected: N/A (No variable assigned)")
            print(f"Slider disabled: No control variable ('{self.slider_control_var_name}') assigned or found.")

    def plot_variable_importance(self):
        """Plots the aggregated variable importance scores."""
        self.vi_ax.clear()
        if self.importance_df is None or self.importance_df.empty:
            self.vi_ax.text(0.5, 0.5, 'No Importance Data Available\n(relative to primary isotope value)', ha='center', va='center', wrap=True, fontsize=9)
            self.vi_canvas.draw()
            return
        try:
            top_n = 15
            df_imp = self.importance_df.copy()
            if 'Feature' not in df_imp.columns: raise ValueError("Importance DataFrame missing 'Feature' column.")
            df_imp['Feature'] = df_imp['Feature'].astype(str)

            def group_feature(f): parts = f.split('_', 1); return parts[0] if len(parts) > 1 and parts[0] else f
            df_imp['GroupedFeature'] = df_imp['Feature'].apply(group_feature)

            grouped_df = df_imp.groupby('GroupedFeature', as_index=False)['Importance'].sum()
            grouped_df = grouped_df[grouped_df['Importance'] > 1e-9]
            plot_data = grouped_df.nlargest(top_n, 'Importance').sort_values(by='Importance', ascending=True)

            if plot_data.empty: self.vi_ax.text(0.5, 0.5, 'No Feature Groups with Importance > 0', ha='center', va='center')
            else:
                self.vi_ax.barh(plot_data['GroupedFeature'], plot_data['Importance'], color='teal')
                self.vi_ax.set_xlabel('Aggregated Importance Score')
                self.vi_ax.set_ylabel('Feature Group')
                # Clarify title that VI is relative to isotope's main value col
                self.vi_ax.set_title(f'Top Feature Groups influencing\n{self.selected_isotope_prefix} (primary value)', fontsize=9)
                self.vi_ax.tick_params(axis='y', labelsize=8); self.vi_ax.tick_params(axis='x', labelsize=8)
                for index, value in enumerate(plot_data['Importance']):
                     self.vi_ax.text(value, index, f' {value:.2E}', va='center', fontsize=7)
                if not plot_data.empty: self.vi_ax.set_xlim(right=plot_data['Importance'].max() * 1.15)

            self.vi_fig.tight_layout()
        except Exception as e:
            print(f"Error plotting variable importance: {e}"); traceback.print_exc()
            self.vi_ax.text(0.5, 0.5, f'Variable Importance Plot Error:\n{e}', ha='center', va='center', color='red', wrap=True)
        self.vi_canvas.draw()

    # --- Main Plotting Method ---
    def plot_dynamic_chart(self):
        """Plots the main dynamic chart based on self.plot_type and scale options."""
        self.dynamic_plot_fig.clf()
        try:
            if self.plot_dimension == '3D': self.dynamic_plot_ax = self.dynamic_plot_fig.add_subplot(111, projection='3d'); print("Created 3D axes.")
            else: self.dynamic_plot_ax = self.dynamic_plot_fig.add_subplot(111); print("Created 2D axes.")
        except Exception as e: print(f"Fatal Error creating axes: {e}"); traceback.print_exc(); QMessageBox.critical(self, "Plot Error", f"Could not create plot axes:\n{e}"); return
        ax = self.dynamic_plot_ax

        if self.isotope_df is None or self.isotope_df.empty:
            ax.text(0.5, 0.5, 'No data loaded for this isotope.', ha='center', va='center', transform=ax.transAxes); self.dynamic_plot_canvas.draw(); return

        # --- Define & Check REQUIRED columns ---

        required_cols = set([self.value_col])
        if self.slider_control_var_name: required_cols.add(self.slider_control_var_name)
        plot_vars = [self.var_x]; required_cols.add(self.var_x) # X always required
        if self.plot_dimension == '3D':
            if self.var_y: plot_vars.append(self.var_y); required_cols.add(self.var_y)
            if self.var_z_or_mode and self.plot_type in ['heatmap_scatter', 'bubble']: plot_vars.append(self.var_z_or_mode); required_cols.add(self.var_z_or_mode)
        if self.var_color and self.plot_type in ['bubble', 'bar3d']: plot_vars.append(self.var_color); required_cols.add(self.var_color)

        required_cols_list = sorted([col for col in required_cols if col]) # Filter None just in case
        missing_cols = [col for col in required_cols_list if col not in self.isotope_df.columns]
        if missing_cols:
            err_msg = f'Plotting Error:\nMissing required columns:\n{", ".join(missing_cols)}'
            print(f"Error: {err_msg}"); ax.text(0.5, 0.5, err_msg, ha='center', va='center', color='red', wrap=True, transform=ax.transAxes); self.dynamic_plot_canvas.draw(); return
        print(f"Required columns for plot confirmed: {required_cols_list}")

        # --- Data Filtering based on CATEGORICAL Slider ---
        df_filtered = self.isotope_df.copy()
        slider_index = self.slider.value(); filter_info = ""; selected_category_for_slider = "All"
        if self.slider_control_var_name and self.slider_categories and self.slider.isEnabled():
             try: # Filter logic remains the same
                 if 0 <= slider_index < len(self.slider_categories):
                     selected_category_for_slider = self.slider_categories[slider_index]
                     df_filtered = df_filtered[df_filtered[self.slider_control_var_name].astype(str) == str(selected_category_for_slider)]
                     filter_info = f'\n(Filtered by "{self.slider_control_var_name}" = "{selected_category_for_slider}")'
                     print(f"Filtering by {self.slider_control_var_name} == '{selected_category_for_slider}'. Points remaining: {len(df_filtered)}")
                 else: df_filtered = df_filtered.iloc[0:0]; filter_info = f'\n(Filter Error: Invalid index {slider_index})'
             except Exception as e: print(f"Error applying slider filter: {e}"); traceback.print_exc(); filter_info = f'\n(Filter Error: {e})'; df_filtered = df_filtered.iloc[0:0]
        elif self.slider_control_var_name and not self.slider.isEnabled(): filter_info = f"\n(Filter '{self.slider_control_var_name}' inactive - Continuous Var)"
        else: filter_info = "\n(No filter applied)"
        if df_filtered.empty:
            msg = f'No data points remaining after filtering.'; # Handle empty df
            if selected_category_for_slider != "All": msg += f'\nFor "{self.slider_control_var_name}" = "{selected_category_for_slider}"'
            print("Plotting aborted: No data after filtering."); ax.text(0.5, 0.5, msg, ha='center', va='center', wrap=True, transform=ax.transAxes); self.dynamic_plot_canvas.draw(); return

        # --- Prepare Data & Norm ---
        plot_df = df_filtered.copy()
        plot_object = None; z_grid_plot = None; surf_df = None
        value_axis_norm = None; norm_vmin = None; norm_vmax = None
        value_data_for_norm = None; plot_title = "Plot"; scale_label_suffix = ""

        try:
            # Convert the selected value column to numeric first
            plot_df[self.value_col] = pd.to_numeric(plot_df[self.value_col], errors='coerce')
            # Drop rows where the PLOTTING target value is NaN
            plot_df.dropna(subset=[self.value_col], inplace=True)
            if plot_df.empty: raise ValueError(f"No valid numeric data found in the selected value column '{self.value_col}'.")

            # --- Configure Norm based on scale options ---
            print(f"Configuring value scale: {self.scale_type}, {self.range_type}")
            scale_label_suffix = f"\n({self.scale_type}, {self.range_type})"
            if self.scale_type == "Logarithmic": value_axis_norm = LogNorm(clip=False)
            else: value_axis_norm = Normalize() # Linear

            if self.range_type == "Fixed Range":
                norm_vmin = self.fixed_min; norm_vmax = self.fixed_max
                if isinstance(value_axis_norm, LogNorm): # Validate fixed log range
                    if norm_vmin is not None and norm_vmin <= 0: norm_vmin = LOG_FLOOR; print(f"Warning: Fixed Min <= 0 for Log scale. Adjusted to {LOG_FLOOR:.1E}")
                    if norm_vmax is not None and norm_vmax <= 0: raise ValueError(f"Fixed Max ({norm_vmax}) <= 0. Cannot use fixed Log scale.")
                    if norm_vmin is not None and norm_vmax is not None and norm_vmax <= norm_vmin: raise ValueError(f"Fixed Min ({norm_vmin}) >= Fixed Max ({norm_vmax}) for Log scale.")
                print(f"Using Fixed Range: Min={norm_vmin}, Max={norm_vmax}")
            else: # Adjust to Data
                value_data_for_norm = plot_df[self.value_col] # Use data after initial NaN drop
                if value_data_for_norm.empty: raise ValueError("No valid data available to adjust range.")
                data_min = value_data_for_norm.min(); data_max = value_data_for_norm.max()
                if isinstance(value_axis_norm, LogNorm):
                     pos_data = value_data_for_norm[value_data_for_norm > 0]
                     if pos_data.empty: raise ValueError("No positive data available for adjusted Log scale.")
                     norm_vmin = max(pos_data.min(), LOG_FLOOR); norm_vmax = pos_data.max()
                else: norm_vmin = data_min; norm_vmax = data_max
                if norm_vmin is None or norm_vmax is None or norm_vmin >= norm_vmax: # Handle single value
                     val = data_min if norm_vmin is not None else 0
                     print(f"Warning: Single value ({val}) or invalid range for 'Adjust to Data'. Creating small range.")
                     if isinstance(value_axis_norm, LogNorm): norm_vmin = max(LOG_FLOOR, val * 0.5); norm_vmax = val * 2.0 if val > 0 else norm_vmin * 100
                     else: norm_vmin = val - 0.5; norm_vmax = val + 0.5
                     if norm_vmin == norm_vmax: norm_vmax += 1.0
                print(f"Adjusted Range Min: {norm_vmin:.3E}, Max: {norm_vmax:.3E}")
                scale_label_suffix = f"\n({self.scale_type}, Adjusted)"

            # Apply determined vmin/vmax to the norm object
            if value_axis_norm is not None:
                 value_axis_norm.vmin = norm_vmin # Can be None if fixed range was only one-sided
                 value_axis_norm.vmax = norm_vmax
                 print(f"Final Norm Applied: vmin={value_axis_norm.vmin}, vmax={value_axis_norm.vmax}")
            else: print("Warning: Value axis norm object not created."); value_axis_norm = Normalize() # Fallback


            # --- Filter Data based on Final Norm Range before plotting ---
            initial_count = len(plot_df)
            if value_axis_norm.vmin is not None: plot_df = plot_df[plot_df[self.value_col] >= value_axis_norm.vmin]
            if value_axis_norm.vmax is not None: plot_df = plot_df[plot_df[self.value_col] <= value_axis_norm.vmax]
            print(f"Points remaining after applying norm range filter: {len(plot_df)} (out of {initial_count})")
            if plot_df.empty: raise ValueError("No data points remain within the specified scale range.")

            # --- Plotting Logic ---
            if self.plot_type == 'bubble': # Unaffected by scale choice for norm
                print("Preparing data for 3D Bubble Chart...")
                # Bubble always requires X, Y, Z, Color, Size(value_col)
                x_codes, x_cats, _ = self._encode_categorical(plot_df, self.var_x)
                y_codes, y_cats, _ = self._encode_categorical(plot_df, self.var_y)
                z_codes, z_cats, _ = self._encode_categorical(plot_df, self.var_z_or_mode)  # Z is var_z_or_mode
                c_codes, c_cats, c_map = self._encode_categorical(plot_df, self.var_color)  # Color is var_color

                plot_df['x_code'] = x_codes;
                plot_df['y_code'] = y_codes;
                plot_df['z_code'] = z_codes;
                plot_df['c_code'] = c_codes
                plot_df['size_numeric'] = pd.to_numeric(plot_df[self.value_col], errors='coerce')

                cols_to_check = ['x_code', 'y_code', 'z_code', 'c_code', 'size_numeric']
                nan_mask = plot_df[cols_to_check].isna().any(axis=1) | (
                            plot_df[['x_code', 'y_code', 'z_code', 'c_code']].fillna(-1) == -1).any(axis=1)
                plot_df = plot_df[~nan_mask]
                if plot_df.empty: raise ValueError("No valid data points remain after NaN check for bubble chart.")

                sizes = plot_df['size_numeric']
                size_min, size_max = sizes.min(), sizes.max()
                size_range = size_max - size_min if size_max > size_min else 1.0
                min_px, max_px = 20, 700
                size_norm = min_px + (max_px - min_px) * (sizes - size_min) / size_range
                size_norm = np.maximum(size_norm, 1.0)  # Min size 1

                plot_object = ax.scatter(plot_df['x_code'], plot_df['y_code'], plot_df['z_code'],
                                         s=size_norm, c=plot_df['c_code'], cmap=self.bubble_cmap, alpha=0.7)

                self._setup_categorical_axis('x', self.var_x, x_cats, ax)
                self._setup_categorical_axis('y', self.var_y, y_cats, ax)
                self._setup_categorical_axis('z', self.var_z_or_mode, z_cats, ax)

                if plot_object and c_cats:
                    handles, _ = plot_object.legend_elements(prop="colors", num=len(c_cats), alpha=0.7)
                    legend_cats = c_cats[:len(handles)]
                    ax.legend(handles, legend_cats, title=self.var_color, loc="center left", bbox_to_anchor=(1.05, 0.5),
                              fontsize=8)

                plot_title = f'3D Bubble: {self.var_x}/{self.var_y}/{self.var_z_or_mode} | Size={self.value_col} | Color={self.var_color} ({len(plot_df)} pts){filter_info}'

            elif self.plot_type == 'bar3d': # Height unaffected, norm not used for bars
                print("Preparing data for 3D Stacked Bar Chart with Fixed Log Scale and Value-Based Stacking...")
                # Bar always requires X, Y, Color(segments), Height(value_col)
                FIXED_VALUE_MIN = 1e-12;
                FIXED_VALUE_MAX = 1e-2
                FIXED_LOG_MIN = np.log10(FIXED_VALUE_MIN);
                FIXED_LOG_MAX = np.log10(FIXED_VALUE_MAX)
                LOG_INPUT_FLOOR = 1e-15

                x_codes, x_cats, x_map = self._encode_categorical(plot_df, self.var_x)
                y_codes, y_cats, y_map = self._encode_categorical(plot_df, self.var_y)
                color_codes, color_cats, color_map = self._encode_categorical(plot_df,
                                                                              self.var_color)  # Color is var_color

                plot_df['x_code'] = x_codes;
                plot_df['y_code'] = y_codes
                plot_df['color_code'] = color_codes
                plot_df['value_numeric'] = pd.to_numeric(plot_df[self.value_col], errors='coerce')
                plot_df['color_category_name'] = plot_df[self.var_color].astype(str)

                cols_to_check = ['x_code', 'y_code', 'color_code', 'value_numeric', 'color_category_name']
                nan_mask = plot_df[cols_to_check].isna().any(axis=1) | (
                            plot_df[['x_code', 'y_code', 'color_code']].fillna(-1) == -1).any(axis=1)
                plot_df_clean = plot_df[~nan_mask].copy()
                if plot_df_clean.empty: raise ValueError(
                    "No valid data points remain after NaN check for stacked bar chart.")

                sorted_category_names_by_value = []
                if 'value_numeric' in plot_df_clean.columns and 'color_category_name' in plot_df_clean.columns:
                    valid_numeric_mask = pd.to_numeric(plot_df_clean['value_numeric'], errors='coerce').notna()
                    if valid_numeric_mask.any():
                        category_means = plot_df_clean[valid_numeric_mask].groupby('color_category_name')[
                            'value_numeric'].mean().sort_values(ascending=True)
                        sorted_category_names_by_value = category_means.index.tolist()
                        print(f"Stacking Order Determined (Lowest Mean to Highest): {sorted_category_names_by_value}")
                    else:
                        print("Warning: No valid numeric values found to determine category stacking order.")
                else:
                    print(f"Warning: Columns for value/category name missing. Cannot determine stacking order.")
                if not sorted_category_names_by_value:
                    sorted_category_names_by_value = sorted(color_cats) if color_cats else []
                    print(f"Warning: Falling back to alphabetical stacking order: {sorted_category_names_by_value}")
                category_dtype = pd.CategoricalDtype(categories=sorted_category_names_by_value, ordered=True)

                agg_funcs = {'value_numeric': 'sum', 'color_category_name': 'first', 'color_code': 'first'}
                grouped = plot_df_clean.groupby(['x_code', 'y_code', 'color_code'], as_index=False).agg(agg_funcs)
                grouped.rename(columns={'value_numeric': 'segment_value'}, inplace=True)
                grouped = grouped[grouped['segment_value'] > 0]
                if grouped.empty: raise ValueError(
                    "No data points with positive summed contributions remain after aggregation.")
                if 'color_category_name' in grouped.columns: grouped['color_category_name'] = grouped[
                    'color_category_name'].astype(category_dtype)

                num_color_cats = len(color_cats) if color_cats else 0
                category_colors = {}
                if num_color_cats > 0:
                    cmap_func = cm.get_cmap(self.bar_cmap)
                    norm_legend = Normalize(vmin=0, vmax=num_color_cats - 1)
                    unique_agg_codes = np.unique(grouped['color_code'].astype(int))
                    for code in unique_agg_codes:
                        if 0 <= code < num_color_cats:
                            category_colors[code] = cmap_func(norm_legend(code))
                        else:
                            category_colors[code] = (0.5, 0.5, 0.5, 0.85)
                print(f"Color Map (Code->Name): {color_map}")
                print(f"Category Colors (Code->RGBA): {category_colors}")

                all_bar_collections = []
                bars_plotted_count = 0
                plot_object = []  # Use list for bars
                for (x_code, y_code), stack_group_orig in grouped.groupby(['x_code', 'y_code']):
                    if 'color_category_name' in stack_group_orig.columns and not stack_group_orig[
                        'color_category_name'].isnull().all():
                        stack_group = stack_group_orig.sort_values('color_category_name', ascending=True)
                    else:
                        print(
                            f"Warning: Could not sort stack ({x_code},{y_code}) by category value. Using default code order.")
                        stack_group = stack_group_orig.sort_values('color_code')
                    stack_group['cum_value'] = stack_group['segment_value'].cumsum()
                    log_stack_base = FIXED_LOG_MIN
                    for index, row in stack_group.iterrows():
                        cumulative_value = row['cum_value']
                        color_code = int(row['color_code'])
                        cum_value_clipped = np.clip(cumulative_value, FIXED_VALUE_MIN, FIXED_VALUE_MAX)
                        log_cum_top = np.log10(np.maximum(cum_value_clipped, LOG_INPUT_FLOOR))
                        segment_zpos = log_stack_base
                        segment_dz = max(log_cum_top - log_stack_base, 0)
                        if segment_dz > 1e-9:
                            segment_color = category_colors.get(color_code, (0.5, 0.5, 0.5, 0.85))
                            dx = dy = 0.7
                            bar_xpos = x_code - dx / 2;
                            bar_ypos = y_code - dy / 2
                            if not np.any(np.isnan([bar_xpos, bar_ypos, segment_zpos, segment_dz])) and \
                                    not np.any(np.isinf([segment_zpos, segment_dz])):
                                coll = ax.bar3d(bar_xpos, bar_ypos, segment_zpos, dx, dy, segment_dz,
                                                color=segment_color, shade=True, alpha=0.85)
                                all_bar_collections.append(coll)  # Keep track if needed
                                plot_object.append(coll)  # Add to the main plot object list
                                bars_plotted_count += 1
                            else:
                                print(
                                    f"Warning: Skipping bar segment due to NaN/Inf value at (x={x_code}, y={y_code}, color={color_code}) zpos={segment_zpos}, dz={segment_dz}")
                        log_stack_base = log_cum_top
                if bars_plotted_count == 0: raise ValueError("No bar segments were plotted.")
                print(
                    f"Plotted {bars_plotted_count} bar segments across {len(grouped.groupby(['x_code', 'y_code']))} locations.")

                self._setup_categorical_axis('x', self.var_x, x_cats, ax)
                self._setup_categorical_axis('y', self.var_y, y_cats, ax)
                ax.set_zlabel(f"{self.value_col} (Log Scale: {FIXED_VALUE_MIN:.0E} to {FIXED_VALUE_MAX:.0E})")
                ax.set_zlim(FIXED_LOG_MIN - 0.5, FIXED_LOG_MAX + 0.5)
                log_ticks = np.arange(np.floor(FIXED_LOG_MIN), np.ceil(FIXED_LOG_MAX) + 1);
                log_ticks = log_ticks[(log_ticks >= FIXED_LOG_MIN) & (log_ticks <= FIXED_LOG_MAX)]

                # Define the formatter function
                def log_tick_formatter(val, pos):
                    return f"$10^{{{int(np.round(val))}}}$"

                ax.set_zticks(log_ticks);
                ax.zaxis.set_major_formatter(FuncFormatter(log_tick_formatter));
                ax.tick_params(axis='z', labelsize=8)

                if color_cats and isinstance(color_map, dict) and category_colors:
                    proxies = [];
                    legend_labels = [];
                    code_to_name_map = {c: name for name, c in color_map.items()}
                    print(f"Attempting legend creation using {len(category_colors)} plotted category colors.")
                    sorted_codes = sorted(category_colors.keys())
                    for code in sorted_codes:
                        proxy_color = category_colors[code];
                        category_name = code_to_name_map.get(code)
                        if category_name is not None:
                            proxy = plt.Rectangle((0, 0), 1, 1, fc=proxy_color, alpha=0.85);
                            proxies.append(proxy);
                            legend_labels.append(str(category_name))
                        else:
                            print(f"Warning: Could not find category name for plotted code {code}.")
                    if proxies and legend_labels:
                        print(f"Creating legend with {len(proxies)} proxies and {len(legend_labels)} labels.")
                        try:
                            ordered_proxies = [];
                            ordered_labels = [];
                            name_to_proxy_map = dict(zip(legend_labels, proxies))
                            for name in sorted_category_names_by_value:
                                if name in name_to_proxy_map: ordered_proxies.append(
                                    name_to_proxy_map[name]); ordered_labels.append(name)
                            if not ordered_proxies: ordered_proxies, ordered_labels = proxies, legend_labels  # Fallback
                            ax.legend(ordered_proxies, ordered_labels, title=f"{self.var_color}\n(Segments)",
                                      loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8)
                        except Exception as legend_err:
                            print(f"ERROR during ax.legend() call: {legend_err}"); traceback.print_exc()
                    else:
                        print("Warning: No valid proxies or labels generated for legend.")
                else:
                    print("Warning: Skipping legend creation due to missing elements.")
                plot_title = f'3D Bar Chart: {self.var_x} vs {self.var_y} | Height={self.value_col} | Segments={self.var_color} ({len(grouped.groupby(["x_code", "y_code"]))} stacks){filter_info}'

            elif self.plot_type == 'heatmap_scatter': # Uses norm for color / Y axis
                plot_title_prefix = "3D Heatmap" if self.plot_dimension == "3D" else "2D Heatmap"
                print(f"Plotting {plot_title_prefix}...")
                # Encode X, Y and Z using the filtered plot_df
                x_codes, x_cats, _ = self._encode_categorical(plot_df, self.var_x, sort_by_value=True, value_col_for_sort=self.value_col)
                plot_df['x_code'] = x_codes
                if self.plot_dimension == '3D':
                    y_codes, y_cats, _ = self._encode_categorical(plot_df, self.var_y, sort_by_value=True, value_col_for_sort=self.value_col)
                    z_codes, z_cats, _ = self._encode_categorical(plot_df, self.var_z_or_mode, sort_by_value=True, value_col_for_sort=self.value_col)
                    plot_df['y_code'] = y_codes; plot_df['z_code'] = z_codes

                # Remove rows where encoding failed (code is -1) AFTER filtering by norm
                code_cols = ['x_code']
                if self.plot_dimension == '3D': code_cols.extend(['y_code', 'z_code'])
                valid_code_mask = (plot_df[code_cols] != -1).all(axis=1)
                if not valid_code_mask.all():
                     print(f"Warning: Removing { (~valid_code_mask).sum() } rows due to encoding errors after filtering.")
                     plot_df = plot_df[valid_code_mask]
                if plot_df.empty: raise ValueError("No data points remain after encoding for heatmap.")

                # Calculate alphas based on final filtered plot_df
                min_alpha, max_alpha = 0.7, 1.0; vals = plot_df[self.value_col]; val_range = vals.max() - vals.min() if vals.max() > vals.min() else 1.0
                alphas = min_alpha + (max_alpha - min_alpha) * (vals - vals.min()) / val_range; alphas = np.clip(alphas, min_alpha, max_alpha)

                value_plot = plot_df[self.value_col]; x_plot = plot_df['x_code']
                if self.plot_dimension == '3D':
                    y_plot = plot_df['y_code']; z_plot = plot_df['z_code']
                    plot_object = ax.scatter(x_plot, y_plot, z_plot, c=value_plot, cmap=self.heatmap_cmap, norm=value_axis_norm, marker='s', s=90, alpha=alphas)
                    try: ax.set_box_aspect(aspect = (np.ptp(x_plot), np.ptp(y_plot), np.ptp(z_plot)))
                    except Exception as aspect_err: print(f"Warning: Could not set 3D box aspect: {aspect_err}")
                    self._setup_categorical_axis('x', self.var_x, x_cats, ax)
                    self._setup_categorical_axis('y', self.var_y, y_cats, ax)
                    self._setup_categorical_axis('z', self.var_z_or_mode, z_cats, ax)
                    plot_title = f'3D Heatmap: {self.var_x} vs {self.var_y} vs {self.var_z_or_mode} | Color={self.value_col} ({len(plot_df)} pts){filter_info}'

                else: # 2D
                    y_plot_2d = value_plot # Y-axis is the chosen value_col
                    plot_object = ax.scatter(x_plot, y_plot_2d, c=value_plot, cmap=self.heatmap_cmap, norm=value_axis_norm, marker='s', s=50, alpha=alphas)
                    self._setup_categorical_axis('x', self.var_x, x_cats, ax)
                    ax.set_ylabel(f"{self.value_col} (Value)")
                    if isinstance(value_axis_norm, LogNorm): ax.set_yscale('log')
                    else: ax.set_yscale('linear')
                    if value_axis_norm.vmin is not None and value_axis_norm.vmax is not None: ax.set_ylim(bottom=value_axis_norm.vmin, top=value_axis_norm.vmax)
                    ax.tick_params(axis='y', labelsize=8)
                    plot_title = f'2D Heatmap: {self.var_x} vs {self.value_col} | Color={self.value_col} ({len(plot_df)} pts){filter_info}'

            elif self.plot_type == 'surface': # Uses norm for height/color
                 print("Plotting Surface plot...")
                 # X and Y types determined from stored lists
                 is_x_cat = self.var_x in self.categorical_variables
                 is_y_cat = self.var_y in self.categorical_variables
                 # Use the already filtered plot_df for surface generation
                 z_values = plot_df[self.value_col] # Use filtered Z values

                 if is_x_cat and is_y_cat:
                    # Re-encode X and Y based only on the filtered data for pivoting
                    x_codes_filt, x_cats_filt, _ = self._encode_categorical(plot_df, self.var_x)
                    y_codes_filt, y_cats_filt, _ = self._encode_categorical(plot_df, self.var_y)
                    surf_df = pd.DataFrame({'x_code': x_codes_filt, 'y_code': y_codes_filt, 'z_value': z_values})
                    surf_df = surf_df[(surf_df['x_code'] != -1) & (surf_df['y_code'] != -1)] # Drop encoding errors
                    if surf_df.empty: raise ValueError("No data left for Cat/Cat surface after filtering/encoding.")

                    pivot = pd.pivot_table(surf_df, values='z_value', index='y_code', columns='x_code', aggfunc=np.mean)
                    x_grid_codes = pivot.columns.values; y_grid_codes = pivot.index.values
                    X_grid_plot, Y_grid_plot = np.meshgrid(x_grid_codes, y_grid_codes)
                    z_grid_plot = pivot.reindex(index=y_grid_codes, columns=x_grid_codes).values

                    plot_object = ax.plot_surface(X_grid_plot, Y_grid_plot, z_grid_plot, cmap=self.surface_cmap, norm=value_axis_norm, rstride=1, cstride=1, linewidth=0.1, antialiased=True, alpha=0.9)
                    self._setup_categorical_axis('x', self.var_x, x_cats_filt, ax) # Use filtered categories
                    self._setup_categorical_axis('y', self.var_y, y_cats_filt, ax)
                    ax.set_zlabel(f"{self.value_col} (Mean)")
                    plot_title = f'Surface: {self.var_x} (Cat) vs {self.var_y} (Cat) | Z={self.value_col} (Mean){filter_info}'

                 elif not is_x_cat and not is_y_cat:

                    x_values = pd.to_numeric(plot_df[self.var_x], errors='coerce')
                    y_values = pd.to_numeric(plot_df[self.var_y], errors='coerce')
                    surf_df = pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values}).dropna() # Drop NaNs from X/Y conversion too
                    if len(surf_df) < 3: raise ValueError(f"Not enough valid data points ({len(surf_df)}) for continuous surface plot interpolation after filtering.")

                    xi = np.linspace(surf_df['x'].min(), surf_df['x'].max(), 100)
                    yi = np.linspace(surf_df['y'].min(), surf_df['y'].max(), 100)
                    X_grid_plot, Y_grid_plot = np.meshgrid(xi, yi)
                    print(f"Interpolating {len(surf_df)} points onto a 100x100 grid...")
                    z_grid_plot = griddata(points=(surf_df['x'], surf_df['y']), values=surf_df['z'], xi=(X_grid_plot, Y_grid_plot), method='linear')
                    print("Interpolation complete.")

                    z_grid_plot_safe = np.nan_to_num(z_grid_plot, nan=np.nanmin(z_grid_plot)) # Use nanmin of original interpolated data
                    plot_object = ax.plot_surface(X_grid_plot, Y_grid_plot, z_grid_plot_safe, cmap=self.surface_cmap, norm=value_axis_norm, rstride=5, cstride=5, linewidth=0, antialiased=False, alpha=0.8)
                    ax.set_xlabel(f"{self.var_x} (Continuous)"); ax.set_ylabel(f"{self.var_y} (Continuous)")
                    ax.set_zlabel(f"{self.value_col} (Interpolated)")
                    ax.xaxis.set_major_locator(LinearLocator(6)); ax.yaxis.set_major_locator(LinearLocator(6)); ax.zaxis.set_major_locator(LinearLocator(6))
                    ax.xaxis.set_major_formatter('{x:.1E}'); ax.yaxis.set_major_formatter('{x:.1E}'); ax.zaxis.set_major_formatter('{x:.1E}')
                    plot_title = f'Surface: {self.var_x} (Cont) vs {self.var_y} (Cont) | Z={self.value_col} (Interpolated){filter_info}'
                 else:
                     # ... (handle mixed type error) ...
                     x_type = "Cat" if is_x_cat else "Cont"; y_type = "Cat" if is_y_cat else "Cont"
                     err_msg = f"Surface plot currently requires both X and Y axes to be the same type.\nSelected: X={x_type}, Y={y_type}"
                     raise ValueError(err_msg) # Raise error to be caught below

            else: raise ValueError(f"Unknown plot type: {self.plot_type}")

            # --- Common Plotting Elements ---
            cbar = None
            if plot_object and self.plot_type in ['heatmap_scatter', 'surface']:
                 cbar_label = f'{self.value_col}{scale_label_suffix}'
                 extend = 'neither'
                 if isinstance(value_axis_norm, LogNorm): extend = 'both'

                 # Check norm range before creating colorbar
                 if value_axis_norm is not None and value_axis_norm.vmin is not None and value_axis_norm.vmax is not None and value_axis_norm.vmin < value_axis_norm.vmax and isinstance(plot_object, plt.cm.ScalarMappable):
                     cbar = self.dynamic_plot_fig.colorbar(plot_object, ax=ax, shrink=0.6, aspect=15, pad=0.15 if self.plot_dimension == '3D' else 0.05, label=cbar_label, norm=value_axis_norm, extend=extend)
                     cbar.ax.tick_params(labelsize=8)
                     if self.plot_type == 'surface': # Format surface bar
                          cbar.locator = LinearLocator(numticks=7); cbar.update_ticks()
                          if abs(value_axis_norm.vmax) > 1e4 or abs(value_axis_norm.vmax) < 1e-2 or (value_axis_norm.vmin is not None and (abs(value_axis_norm.vmin) > 1e4 or abs(value_axis_norm.vmin) < 1e-2)): cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.1E}')
                          else: cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.2f}')
                          cbar.update_ticks()
                 else: print(f"Skipping colorbar creation due to invalid range or plot object type (vmin={value_axis_norm.vmin}, vmax={value_axis_norm.vmax}).")

            ax.set_title(plot_title, fontsize=10, wrap=True)
            if self.plot_type != 'surface': ax.grid(True, linestyle='--', alpha=0.6)

            has_external_element = bool(cbar) or (self.plot_type == 'bubble' and plot_object and 'handles' in locals() and handles) or (self.plot_type == 'bar3d') # Check bubble legend created
            right_margin = 0.82 if has_external_element else 0.98
            try: self.dynamic_plot_fig.subplots_adjust(left=0.1, right=right_margin, bottom=0.15, top=0.90)
            except Exception as layout_err: print(f"Warning applying layout adjustments: {layout_err}")

            self.dynamic_plot_canvas.draw()
            print("Dynamic plot rendering complete.")

        except ValueError as ve: # Catch specific errors like no data, bad ranges
            err_msg = f'Plot Generation Error:\n{ve}'; print(f"Error during plot generation: {ve}");
            try: ax.text(0.5, 0.5, err_msg, ha='center', va='center', color='orange', wrap=True, transform=ax.transAxes); self.dynamic_plot_canvas.draw()
            except Exception as e_inner: print(f"Could not display error on plot canvas: {e_inner}")
        except Exception as e: # Catch other unexpected errors
            err_msg = f'Plot Generation Error:\n{e}'; print(f"Error during plot generation: {e}"); traceback.print_exc()
            try: ax.text(0.5, 0.5, err_msg, ha='center', va='center', color='red', wrap=True, transform=ax.transAxes); self.dynamic_plot_canvas.draw()
            except Exception as e_inner: print(f"Could not display error on plot canvas: {e_inner}")


# --- MainWindow Class ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NWS Project GUI")
        self.setGeometry(100, 100, 500, 680) # Adjusted height
        self.df = None
        self.variables = [] # Element variables (non-isotope, non-target, non-total)
        self.categorical_vars = [] # Subset of self.variables
        self.continuous_vars = [] # Subset of self.variables
        self.isotopes = [] # List of base isotope names
        self.isotope_dfs = {} # Maps base isotope name to its DataFrame
        self.variable_importance = None
        self.value_col_map = {} # Map base isotope name to its default value column name
        self.potential_target_cols = [] # List of ALL numeric cols found
        self.all_isotope_cols_map = {} # Map base isotope name to list of all associated columns

        self.chart_window = None
        self.unused_category_filters_widgets = {}  # To hold dynamically created combos {var_name: combo_box}


        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(); central_widget.setLayout(main_layout)

        # --- Upload Row ---
        upload_layout = QHBoxLayout(); self.upload_button = QPushButton("Upload CSV"); self.upload_button.clicked.connect(self.load_csv); upload_layout.addWidget(self.upload_button); self.file_label = QLabel("No file loaded"); self.file_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); upload_layout.addWidget(self.file_label); main_layout.addLayout(upload_layout)
        # --- Isotope Selection ---
        isotope_layout = QHBoxLayout(); self.isotope_label = QLabel("Isotope Context:"); isotope_layout.addWidget(self.isotope_label); self.isotope_combo = QComboBox(); self.isotope_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed); self.isotope_combo.addItem("Load CSV first"); self.isotope_combo.setEnabled(False); self.isotope_combo.currentIndexChanged.connect(self._on_isotope_change); isotope_layout.addWidget(self.isotope_combo); main_layout.addLayout(isotope_layout)

        # --- Target Variable Selection ---
        target_var_layout = QHBoxLayout()
        target_var_layout.addWidget(QLabel("Target Variable (for Plotting):"))
        self.target_variable_combo = QComboBox()
        self.target_variable_combo.addItem("Load CSV first")
        self.target_variable_combo.setEnabled(False)
        self.target_variable_combo.setToolTip("Select the numeric column to use for the value axis/color/size.")
        target_var_layout.addWidget(self.target_variable_combo)
        main_layout.addLayout(target_var_layout)
        # --- End Target Variable ---

        # --- Plot Type Selection ---
        plot_type_layout = QHBoxLayout(); plot_type_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox(); self.plot_type_combo.addItems(["3D Heatmap", "2D Heatmap", "3D Bubble Chart", "3D Bar Chart", "Surface Plot"]); self.plot_type_combo.setEnabled(False); self.plot_type_combo.currentTextChanged.connect(self._update_variable_combos_state); plot_type_layout.addWidget(self.plot_type_combo); main_layout.addLayout(plot_type_layout)

        # --- Scaling Options ---
        scale_options_layout = QVBoxLayout()
        scale_type_layout = QHBoxLayout(); scale_type_layout.addWidget(QLabel("Value Scale Type:")); self.scale_type_combo = QComboBox(); self.scale_type_combo.addItems(["Linear", "Logarithmic"]); self.scale_type_combo.setEnabled(False); self.scale_type_combo.setToolTip("Choose Linear or Logarithmic scaling."); self.scale_type_combo.currentTextChanged.connect(self._on_scale_range_change); scale_type_layout.addWidget(self.scale_type_combo); scale_options_layout.addLayout(scale_type_layout)
        range_type_layout = QHBoxLayout(); range_type_layout.addWidget(QLabel("Value Scale Range:")); self.range_type_combo = QComboBox(); self.range_type_combo.addItems(["Adjust to Data", "Fixed Range"]); self.range_type_combo.setEnabled(False); self.range_type_combo.setToolTip("Use data range or fixed values."); self.range_type_combo.currentTextChanged.connect(self._on_scale_range_change); range_type_layout.addWidget(self.range_type_combo); scale_options_layout.addLayout(range_type_layout)
        fixed_range_layout = QHBoxLayout(); fixed_range_layout.addWidget(QLabel("Fixed Min:")); self.fixed_min_input = QLineEdit(); self.fixed_min_input.setPlaceholderText("e.g., 1e-12 or 0.1"); self.fixed_min_input.setEnabled(False); self.fixed_min_input.setValidator(QDoubleValidator()); fixed_range_layout.addWidget(self.fixed_min_input); fixed_range_layout.addWidget(QLabel("Fixed Max:")); self.fixed_max_input = QLineEdit(); self.fixed_max_input.setPlaceholderText("e.g., 1e-2 or 100.0"); self.fixed_max_input.setEnabled(False); self.fixed_max_input.setValidator(QDoubleValidator()); fixed_range_layout.addWidget(self.fixed_max_input); scale_options_layout.addLayout(fixed_range_layout)
        main_layout.addLayout(scale_options_layout)

        # --- Variable Selection ---
        self.var_label = QLabel("Select Plot Element Variables:"); main_layout.addWidget(self.var_label)
        self.var_combo1 = QComboBox(); self.var_combo1.addItem("Load CSV first"); self.var_combo1.setEnabled(False)
        self.var_combo2 = QComboBox(); self.var_combo2.addItem("Load CSV first"); self.var_combo2.setEnabled(False)
        self.var_combo3 = QComboBox(); self.var_combo3.addItem("Load CSV first"); self.var_combo3.setEnabled(False)
        self.var_combo4 = QComboBox(); self.var_combo4.addItem("Load CSV first"); self.var_combo4.setEnabled(False)
        var1_layout = QHBoxLayout(); self.var1_label = QLabel("Axis 1 (X):"); var1_layout.addWidget(self.var1_label); var1_layout.addWidget(self.var_combo1); main_layout.addLayout(var1_layout)
        var2_layout = QHBoxLayout(); self.var2_label = QLabel("Var 2:"); var2_layout.addWidget(self.var2_label); var2_layout.addWidget(self.var_combo2); main_layout.addLayout(var2_layout)
        var3_layout = QHBoxLayout(); self.var3_label = QLabel("Var 3:"); var3_layout.addWidget(self.var3_label); var3_layout.addWidget(self.var_combo3); main_layout.addLayout(var3_layout)
        var4_layout = QHBoxLayout(); self.var4_label = QLabel("Var 4:"); var4_layout.addWidget(self.var4_label); var4_layout.addWidget(self.var_combo4); main_layout.addLayout(var4_layout)

        # --- Placeholder for Dynamic Filters ---
        self.filters_groupbox = QGroupBox("Filter Unused Categorical Variables")
        self.filters_layout = QVBoxLayout()
        self.filters_groupbox.setLayout(self.filters_layout)
        self.filters_groupbox.setVisible(False)  # Initially hidden
        main_layout.addWidget(self.filters_groupbox)

        # --- Display Button ---
        self.display_button = QPushButton("Display Chart"); self.display_button.clicked.connect(self.display_chart); self.display_button.setEnabled(False); main_layout.addWidget(self.display_button)
        main_layout.addStretch()

        # --- Connect Signals for Dynamic Filters ---
        self.plot_type_combo.currentTextChanged.connect(self._update_unused_variable_filters_ui)
        self.isotope_combo.currentIndexChanged.connect(
            self._update_unused_variable_filters_ui)  # Update on isotope change too
        self.var_combo1.currentIndexChanged.connect(self._update_unused_variable_filters_ui)
        self.var_combo2.currentIndexChanged.connect(self._update_unused_variable_filters_ui)
        self.var_combo3.currentIndexChanged.connect(self._update_unused_variable_filters_ui)
        self.var_combo4.currentIndexChanged.connect(self._update_unused_variable_filters_ui)

        self._update_variable_combos_state()

    def _clear_layout(self, layout):
        """Recursively clears all widgets and sub-layouts from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)  # Remove item from layout
                widget = item.widget()
                if widget is not None:
                    # If it's a widget, schedule its deletion
                    widget.deleteLater()
                else:
                    # If it's a layout, recursively clear it
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout)

    def _on_isotope_change(self):
        """Update the target variable combo when the isotope context changes."""
        if not self.potential_target_cols or self.isotope_combo.count() == 0:
            return # Not ready yet

        selected_isotope = self.isotope_combo.currentText() # This is the BASE isotope name
        # Find the default value column mapped to this base isotope
        default_target = self.value_col_map.get(selected_isotope)

        # Block signals temporarily to prevent cascade updates if needed
        self.target_variable_combo.blockSignals(True)
        # Find the index of the default target and set it
        try:
             if default_target and default_target in self.potential_target_cols:
                  default_index = self.potential_target_cols.index(default_target)
                  self.target_variable_combo.setCurrentIndex(default_index)
                  print(f"Isotope changed to '{selected_isotope}', default target set to '{default_target}'")
             elif self.target_variable_combo.count() > 0: # Select first item if default not found
                  self.target_variable_combo.setCurrentIndex(0)
                  print(f"Isotope changed to '{selected_isotope}', default target '{default_target}' not found in list, set to '{self.target_variable_combo.currentText()}'")
             else:
                  print(f"Isotope changed to '{selected_isotope}', but no targets available.")
        except ValueError: # Should not happen if list is populated correctly
             if self.target_variable_combo.count() > 0:
                  self.target_variable_combo.setCurrentIndex(0)
             print(f"Isotope changed to '{selected_isotope}', ValueError finding default index, set to '{self.target_variable_combo.currentText()}'")
        self.target_variable_combo.blockSignals(False)


    def _on_scale_range_change(self):
        """Callback to handle changes in scale/range type combos."""
        # (No changes needed)
        is_fixed = self.range_type_combo.currentText() == "Fixed Range"
        data_loaded = self.range_type_combo.isEnabled() # Check if combos are active
        self.fixed_min_input.setEnabled(is_fixed and data_loaded)
        self.fixed_max_input.setEnabled(is_fixed and data_loaded)

    def _update_variable_combos_state(self):
        """Enable/disable/relabel variable combo boxes based on selected plot type and data loaded."""

        selected_plot_type = self.plot_type_combo.currentText()
        is_heatmap_3d = selected_plot_type == "3D Heatmap"
        is_heatmap_2d = selected_plot_type == "2D Heatmap"
        is_bubble = selected_plot_type == "3D Bubble Chart"
        is_bar3d = selected_plot_type == "3D Bar Chart"
        is_surface = selected_plot_type == "Surface Plot"
        is_data_ready = bool(self.df is not None and self.variables) # Relies on self.variables
        self.var_combo1.setEnabled(is_data_ready)
        self.var_combo2.setEnabled(is_data_ready and (is_heatmap_3d or is_heatmap_2d or is_bubble or is_bar3d or is_surface))
        self.var_combo3.setEnabled(is_data_ready and (is_heatmap_3d or is_bubble or is_bar3d))
        self.var_combo4.setEnabled(is_data_ready and (is_heatmap_3d or is_heatmap_2d or is_bubble or is_bar3d or is_surface))
        self.var1_label.setText("Axis 1 (X):")
        self.var2_label.setText("Var 2 (Slider?):"); self.var3_label.setText("Var 3 (N/A):"); self.var4_label.setText("Var 4 (Slider?):") # Defaults
        self._update_unused_variable_filters_ui()
        if is_heatmap_3d: self.var2_label.setText("Axis 2 (Y):"); self.var3_label.setText("Axis 3 (Z):"); self.var4_label.setText("Slider Variable:")
        elif is_heatmap_2d: self.var2_label.setText("Var 2 (Slider?):"); self.var3_label.setText("Var 3 (N/A):"); self.var4_label.setText("Var 4 (Slider?):")
        elif is_bubble: self.var2_label.setText("Axis 2 (Y):"); self.var3_label.setText("Axis 3 (Z):"); self.var4_label.setText("Color:")
        elif is_bar3d: self.var2_label.setText("Axis 2 (Y):"); self.var3_label.setText("Color (Segments):"); self.var4_label.setText("Slider Variable:")
        elif is_surface: self.var2_label.setText("Axis 2 (Y):"); self.var3_label.setText("Var 3 (N/A):"); self.var4_label.setText("Slider Variable:")


    def load_csv(self):

        file_name, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
        if not file_name: self.file_label.setText('File selection cancelled.'); return
        try:
            print(f"Attempting to load CSV: {file_name}")
            try: self.df = pd.read_csv(file_name, low_memory=False)
            except UnicodeDecodeError: print("UTF-8 decoding failed, trying latin1..."); self.df = pd.read_csv(file_name, low_memory=False, encoding='latin1')
            except Exception as read_err: raise Exception(f"Failed to read CSV: {read_err}")
            self.file_label.setText(f'Loaded: {file_name.split("/")[-1]} ({len(self.df)} rows)'); print("CSV Loaded Successfully.")

            # Reset (including new all_isotope_cols_map)
            self.variables = []; self.categorical_vars = []; self.continuous_vars = []
            self.isotopes = []; self.isotope_dfs = {}; self.variable_importance = None
            self.value_col_map = {}; self.potential_target_cols = []
            self.all_isotope_cols_map = {}

            # Clear combos
            self.isotope_combo.clear(); self.target_variable_combo.clear()
            for combo in [self.var_combo1, self.var_combo2, self.var_combo3, self.var_combo4]: combo.clear()

            # --- Step 1: Preprocessing
            self.preprocess_data()

            # Keep track of originally identified isotopes before checking for NaN values
            initial_isotopes_found = self.isotopes[:] # Uses self.isotopes populated by preprocess_data

            # Step 2: Create DataFrames and check for all-NaNs in DEFAULT value column
            dropped_nan_isotopes = []
            if self.df is not None and initial_isotopes_found:
                 print(f"Preprocessing identified {len(initial_isotopes_found)} potential base isotopes. Now creating DataFrames...")
                 dropped_nan_isotopes = self.create_isotope_dataframes() # Returns list of skipped BASE isotopes

                 # Step 3: Update self.isotopes to ONLY include successfully loaded ones
                 self.isotopes = sorted(list(self.isotope_dfs.keys()))
                 print(f"Final base isotopes available after NaN check ({len(self.isotopes)}): {self.isotopes}")

                 # Step 4: Show the alert message for skipped isotopes
                 if dropped_nan_isotopes:
                     message = "The following base isotopes were skipped (default value column all NaN):\n- " + "\n- ".join(dropped_nan_isotopes)
                     QMessageBox.warning(self, "Isotopes Skipped", message)

                 # Step 5: Populate isotope combo box with the *final, filtered* list of BASE isotope names
                 self.isotope_combo.clear()
                 if self.isotopes: self.isotope_combo.addItems(self.isotopes)
                 else: self.isotope_combo.addItem("No Isotopes Loaded"); print("Warning: No isotopes remained after filtering.")

                 # Step 5b: Populate Target Variable Combo with ALL potential numeric columns
                 self.target_variable_combo.clear()
                 if self.potential_target_cols:
                      self.target_variable_combo.addItems(self.potential_target_cols)
                      self._on_isotope_change() # Set default target based on selected isotope (and value_col_map)
                      self.target_variable_combo.setEnabled(True)
                 else:
                      self.target_variable_combo.addItem("No Numeric Cols Found")
                      self.target_variable_combo.setEnabled(False)

            else: # Preprocessing failed to find isotopes
                print("Preprocessing did not identify any isotopes or failed. Skipping DataFrame creation.")
                self.isotope_combo.clear(); self.isotope_combo.addItem("No Isotopes Found")
                self.target_variable_combo.clear(); self.target_variable_combo.addItem("Load CSV first"); self.target_variable_combo.setEnabled(False)

            # --- Step 6: Enable UI based on final state ---
            data_loaded = bool(
                self.df is not None and self.isotope_dfs and self.variables and self.potential_target_cols)

            self.display_button.setEnabled(data_loaded)
            self.isotope_combo.setEnabled(data_loaded and bool(self.isotopes))  # Enable only if there are isotopes
            self.plot_type_combo.setEnabled(data_loaded)
            self.scale_type_combo.setEnabled(data_loaded)
            self.range_type_combo.setEnabled(data_loaded)

            # --- Set Default Scaling Options ---
            if data_loaded:
                print("Setting default scaling options...")
                self.scale_type_combo.blockSignals(True); self.range_type_combo.blockSignals(True)
                self.fixed_min_input.blockSignals(True); self.fixed_max_input.blockSignals(True)
                self.scale_type_combo.setCurrentText("Logarithmic")
                self.range_type_combo.setCurrentText("Fixed Range")
                self.fixed_min_input.setText(f"{FIXED_COLOR_MIN_DEFAULT:.1E}")
                self.fixed_max_input.setText(f"{FIXED_COLOR_MAX_DEFAULT:.1E}")
                self.scale_type_combo.blockSignals(False); self.range_type_combo.blockSignals(False)
                self.fixed_min_input.blockSignals(False); self.fixed_max_input.blockSignals(False)
                self._on_scale_range_change()
                print(f"Defaults set: Scale={self.scale_type_combo.currentText()}, Range={self.range_type_combo.currentText()}, Min={self.fixed_min_input.text()}, Max={self.fixed_max_input.text()}")
            else:
                self._on_scale_range_change()

            # Update variable combo states
            self._update_variable_combos_state()

            if not data_loaded:
                 msg = "Could not identify necessary data (isotopes, variables, or targets) after processing."

                 if self.df is not None:
                     if not initial_isotopes_found: msg = "No columns matching the isotope pattern found."
                     elif not self.isotope_dfs: msg = "Isotopes found, but failed to load data for any (check logs/skipped message)."
                     elif not self.variables: msg = "Isotope data loaded, but failed to identify usable element variables."
                     elif not self.potential_target_cols: msg = "Isotope/variable data loaded, but failed to identify numeric target columns."
                 QMessageBox.warning(self, "Processing Issue", msg); print(f"Processing Issue: {msg}. UI may be disabled.")
            else:
                print("Processing complete. UI enabled with loaded data.")

        except Exception as e:

            error_message = f"Failed to load or process CSV:\n{e}"; self.file_label.setText(f'Error loading file.'); print(f"Error during CSV load/process: {e}"); traceback.print_exc(); QMessageBox.critical(self, "Load Error", error_message)
            # Reset everything on error
            self.df = None; self.variables = []; self.categorical_vars = []; self.continuous_vars = []
            self.isotopes = []; self.isotope_dfs = {}; self.value_col_map = {}; self.potential_target_cols = []
            self.all_isotope_cols_map = {}
            self.isotope_combo.clear(); self.target_variable_combo.clear(); self.var_combo1.clear(); self.var_combo2.clear(); self.var_combo3.clear(); self.var_combo4.clear();
            self.isotope_combo.addItem("Load CSV first"); self.target_variable_combo.addItem("Load CSV first")
            for combo in [self.var_combo1, self.var_combo2, self.var_combo3, self.var_combo4]: combo.addItem("Load CSV first")
            self.display_button.setEnabled(False); self.isotope_combo.setEnabled(False); self.target_variable_combo.setEnabled(False); self.plot_type_combo.setEnabled(False);
            self.scale_type_combo.setEnabled(False); self.range_type_combo.setEnabled(False); self.fixed_min_input.setEnabled(False); self.fixed_max_input.setEnabled(False)
            self._update_variable_combos_state()


    def _is_numeric_col(self, col_name):
        """Helper to check if a column is likely numeric based on a sample."""
        if self.df is None or col_name not in self.df.columns:
            return False
        try:
            # Sample non-NA values for efficiency
            sample = self.df.loc[self.df[col_name].notna(), col_name].head(50)
            if sample.empty: # Handle columns that might be all NA
                 # Check dtype directly if sample is empty
                 return pd.api.types.is_numeric_dtype(self.df[col_name])
            # Attempt conversion on the sample
            return pd.to_numeric(sample, errors='coerce').notna().any()
        except Exception as e:
            print(f"  - Warning: Error checking numeric status for '{col_name}': {e}")
            return False

    def preprocess_data(self):
        """
        Identifies variables, isotopes (using regex), types, potential targets,
        and populates initial lists. Groups isotope columns like X-123 and X-123.1.
        Variables are non-isotope, non-total, non-case columns.
        Targets are all numeric columns.
        """
        if self.df is None:
            print("Preprocessing skipped: DataFrame is not loaded."); return
        print("Starting preprocessing with Regex Isotope Identification...")
        try:
            all_columns = self.df.columns.tolist()
            if not all_columns: raise ValueError("CSV file appears to have no columns.")

            # --- Initialization ---
            potential_isotopes_cols = {} # Maps base_isotope_name -> [full_col_name1, ...]
            non_isotope_or_derivative_cols = [] # Temp list for non-isotope pattern matches
            self.potential_target_cols = [] # Final list of all numeric cols
            self.isotopes = []              # Final list of base isotope names
            self.value_col_map = {}         # Final map of base_isotope -> default_value_col
            self.all_isotope_cols_map = {}  # Final map of base_isotope -> [all_related_cols]
            self.variables = []             # Final list of element variable names
            self.categorical_vars = []      # Final list of categorical element variables
            self.continuous_vars = []       # Final list of continuous element variables

            print("Scanning all columns for isotopes and potential targets...")
            for col_name in all_columns:
                if not isinstance(col_name, str) or not col_name.strip():
                    print(f"  - Skipping non-string or empty column name: {col_name}")
                    continue

                col_name_str = col_name.strip()

                # --- Check if column is numeric (for potential targets) ---
                if self._is_numeric_col(col_name_str):
                    if col_name_str not in self.potential_target_cols: # Avoid duplicates
                        self.potential_target_cols.append(col_name_str)

                # --- Check for Isotope Pattern Match ---
                match = ISOTOPE_PATTERN.match(col_name_str)
                is_isotope_match = bool(match)
                base_isotope_name = None

                if is_isotope_match:
                    # Try splitting by '.' to find base name
                    parts = col_name_str.split('.', 1)
                    base_candidate = parts[0]
                    base_match = ISOTOPE_BASE_PATTERN.match(base_candidate)

                    if base_match and base_match.group(1) == base_candidate:
                        base_isotope_name = base_candidate # Found valid base before '.'
                    else:
                        # Check if the full name itself is a base pattern
                        base_match_full = ISOTOPE_BASE_PATTERN.match(col_name_str)
                        if base_match_full and base_match_full.group(1) == col_name_str:
                            base_isotope_name = col_name_str # Full name is the base


                    if base_isotope_name:
                        if base_isotope_name not in potential_isotopes_cols:
                            potential_isotopes_cols[base_isotope_name] = []
                        potential_isotopes_cols[base_isotope_name].append(col_name_str)

                    else:
                        # Matched pattern but couldn't find base - add to non-isotope list
                        non_isotope_or_derivative_cols.append(col_name_str)

                else:
                    # Did not match isotope pattern - add to non-isotope list
                    non_isotope_or_derivative_cols.append(col_name_str)


            # --- Finalize Potential Targets ---
            self.potential_target_cols = sorted(list(set(self.potential_target_cols))) # Unique and sorted
            print(f"Found {len(self.potential_target_cols)} potential numeric target columns.")

            # --- Determine Final Isotopes and Default Value Columns ---
            self.isotopes = sorted(list(potential_isotopes_cols.keys()))
            self.all_isotope_cols_map = potential_isotopes_cols # Store the full map
            print(f"Identified {len(self.isotopes)} base isotopes: {self.isotopes}")
            print("Determining default value column for each base isotope...")
            for base_name in self.isotopes:
                assoc_cols = self.all_isotope_cols_map[base_name]
                default_value_col = None
                # Priority 1: Exact match for base name, if numeric
                if base_name in assoc_cols and self._is_numeric_col(base_name):
                    default_value_col = base_name
                    print(f"  - Base '{base_name}': Using exact match '{default_value_col}' as default (numeric).")
                # Priority 2: First associated column that is numeric
                if default_value_col is None:
                    for col in assoc_cols:
                        if self._is_numeric_col(col):
                            default_value_col = col
                            print(f"  - Base '{base_name}': Using first numeric associated column '{default_value_col}' as default.")
                            break
                # Priority 3: First associated column (fallback)
                if default_value_col is None:
                    default_value_col = assoc_cols[0]
                    print(f"  - Base '{base_name}': No numeric found. Using first associated column '{default_value_col}' as default (fallback).")

                self.value_col_map[base_name] = default_value_col

            # --- Identify Final Element Variables and Types ---
            # Start from the list of columns that weren't isotopes/derivatives
            # and filter out 'total'/'case'
            print("Identifying final element variables (non-isotope, non-total, non-case)...")
            potential_element_vars = []
            for col_name in non_isotope_or_derivative_cols:
                # Ensure it's a string before checking content
                if not isinstance(col_name, str):
                    continue # Skip non-string column names (already handled, but safe)

                # Explicitly exclude columns containing 'total' or 'case' (case-insensitive)
                col_lower = col_name.strip().lower()
                if 'total' in col_lower or 'case' in col_lower:
                    print(f"  - Skipping potential variable '{col_name}' (contains 'total' or 'case').")
                    continue

                # If it passed the checks, add it as a potential element variable
                potential_element_vars.append(col_name)

            # Determine types for these filtered potential element variables
            print(f"Determining types for {len(potential_element_vars)} potential element variables...")
            for var_col in potential_element_vars:
                if var_col in self.df.columns: # Double-check existence
                    try:
                        # --- Type determination logic ---
                        n_unique = self.df[var_col].nunique(dropna=False)
                        if n_unique < CATEGORICAL_THRESHOLD:
                            print(f"  - '{var_col}' -> Categorical ({n_unique} unique).")
                            self.categorical_vars.append(var_col)
                            self.variables.append(var_col)
                            # Ensure consistency, convert to string if categorical
                            if not pd.api.types.is_string_dtype(self.df[var_col]):
                                 self.df[var_col] = self.df[var_col].astype(str)
                        else:
                            # Use _is_numeric_col helper for continuous check
                            if self._is_numeric_col(var_col):
                                print(f"  - '{var_col}' -> Continuous ({n_unique} unique).")
                                self.continuous_vars.append(var_col)
                                self.variables.append(var_col)
                                # Ensure stored as numeric after check
                                self.df[var_col] = pd.to_numeric(self.df[var_col], errors='coerce')
                            else:
                                # Treat as categorical if not convertible to numeric
                                print(f"  - '{var_col}' -> Categorical ({n_unique} unique, non-numeric).")
                                self.categorical_vars.append(var_col)
                                self.variables.append(var_col)
                                if not pd.api.types.is_string_dtype(self.df[var_col]):
                                     self.df[var_col] = self.df[var_col].astype(str)

                    except Exception as e:
                        print(f"  - Error processing column '{var_col}' for type: {e}. Skipping.")
                else:
                    print(f"  - Warning: Potential variable '{var_col}' disappeared from DataFrame?")

            # Sort the final variable lists
            self.variables = sorted(self.variables)
            self.categorical_vars = sorted(self.categorical_vars)
            self.continuous_vars = sorted(self.continuous_vars)

            print(f"Final Usable Element Variables ({len(self.variables)}): {self.variables}")
            print(f"  Categorical ({len(self.categorical_vars)}): {self.categorical_vars}")
            print(f"  Continuous ({len(self.continuous_vars)}): {self.continuous_vars}")

            # --- Populate Variable Combo Boxes ---
            for combo in [self.var_combo1, self.var_combo2, self.var_combo3, self.var_combo4]:
                combo.clear()
            no_vars_msg = "No Element Variables Found"
            if self.variables:
                 for combo in [self.var_combo1, self.var_combo2, self.var_combo3, self.var_combo4]:
                     combo.addItems(self.variables)
                 # Set defaults if variables exist
                 num_vars = len(self.variables)
                 if num_vars > 0: self.var_combo1.setCurrentIndex(0)
                 if num_vars > 1: self.var_combo2.setCurrentIndex(min(1, num_vars - 1))
                 if num_vars > 2: self.var_combo3.setCurrentIndex(min(2, num_vars - 1))
                 if num_vars > 3: self.var_combo4.setCurrentIndex(min(3, num_vars - 1))
            else:
                 for combo in [self.var_combo1, self.var_combo2, self.var_combo3, self.var_combo4]:
                     combo.addItem(no_vars_msg)

            print("Preprocessing finished.")

        except Exception as e:
            print(f"Error during preprocessing:\n{e}"); traceback.print_exc()
            # Ensure lists are cleared on error
            self.variables = []; self.categorical_vars = []; self.continuous_vars = []
            self.isotopes = []; self.value_col_map = {}; self.potential_target_cols = []
            self.all_isotope_cols_map = {}
            raise # Re-raise the exception to be caught by load_csv

    def create_isotope_dataframes(self):
        """
        Creates DFs for isotopes identified in self.isotopes (base names),
        including all related columns.
        Skips those where the default value col is all-NaN.
        Includes all element variables and potential target columns.
        """
        if self.df is None or not self.isotopes:
            print("Cannot create isotope DataFrames: No data or no isotopes identified."); return []
        print(f"Attempting to create DataFrames for {len(self.isotopes)} identified base isotopes...")
        self.isotope_dfs = {} # Reset
        dropped_isotopes_nan = []
        base_element_vars_to_include = self.variables # Use final element variables
        all_potential_targets_to_include = self.potential_target_cols # Include all numeric cols

        for base_iso_name in self.isotopes:
            # Get all columns associated with this base isotope
            related_isotope_cols = self.all_isotope_cols_map.get(base_iso_name, [])
            default_value_col = self.value_col_map.get(base_iso_name) # Get the pre-determined default

            if not default_value_col:
                 print(f"Warning: No default value column mapped for base isotope '{base_iso_name}'. Skipping DF creation.")
                 continue
            if default_value_col not in self.df.columns:
                 print(f"Warning: Mapped default value column '{default_value_col}' not found in DataFrame for base isotope '{base_iso_name}'. Skipping DF creation.")
                 continue

            # Combine all necessary columns: base element vars + related isotope cols + all potential targets
            cols_for_this_isotope = set(base_element_vars_to_include) | set(related_isotope_cols) | set(all_potential_targets_to_include)
            # Ensure all selected columns actually exist in the original DataFrame
            cols_selection = sorted([col for col in cols_for_this_isotope if col in self.df.columns])

            if not cols_selection:
                print(f"Error: No valid columns found for base isotope '{base_iso_name}' after combining. Skipping.")
                continue

            try:
                # Create the initial DataFrame slice
                iso_df = self.df[cols_selection].copy()

                # --- Check default value_col for all NaN ---
                # This check uses the default column identified in preprocess_data
                if iso_df[default_value_col].isnull().all():
                    print(f"Warning: Skipping base isotope '{base_iso_name}' because its DEFAULT value column ('{default_value_col}') is all NaN.")
                    dropped_isotopes_nan.append(base_iso_name)
                    continue # Skip to the next isotope

                # --- Imputation (apply to the valid DataFrame) ---
                # Impute numeric columns (includes targets and continuous element vars)
                numeric_cols_in_iso_df = iso_df.select_dtypes(include=np.number).columns
                for col in numeric_cols_in_iso_df:
                    if iso_df[col].isnull().any():
                        # Impute all numeric NaNs with a small floor
                        fill_val = LOG_FLOOR

                        iso_df[col] = iso_df[col].fillna(fill_val)
                        # print(f"  - Imputed NaNs in numeric column '{col}' for '{base_iso_name}' with {fill_val}") # Optionally you can get it to print when columns have been imputed (too many on the example to have this enabled)
                # Impute categorical columns (element variables identified as categorical)
                for col in self.categorical_vars: # Use the list of identified categorical element vars
                     if col in iso_df.columns and iso_df[col].isnull().any():
                         iso_df[col] = iso_df[col].fillna('Missing')


                # Store the processed DataFrame, mapped by the base isotope name
                self.isotope_dfs[base_iso_name] = iso_df
                # print(f"Successfully created and processed DataFrame for base isotope: '{base_iso_name}'") # Verbose

            except Exception as e:
                print(f"Error creating/processing DF for base isotope '{base_iso_name}': {e}")
                traceback.print_exc()

        final_isotope_count = len(self.isotope_dfs)
        print(f"Finished creating isotope DataFrames. {final_isotope_count} successfully created.")
        return dropped_isotopes_nan # Return list of skipped base names



    def _update_unused_variable_filters_ui(self):
        """
        Dynamically creates ComboBoxes for filtering unused categorical vars.
        Defaults the selection to the first actual category value if available.
        """
        # --- Use the robust clearing function ---

        self._clear_layout(self.filters_layout)
        self.unused_category_filters_widgets = {} # Reset stored widgets


        # --- Check if data and necessary selections are ready ---
        if self.df is None or not self.categorical_vars or self.isotope_combo.currentIndex() < 0:
            self.filters_groupbox.setVisible(False)
            return

        selected_isotope = self.isotope_combo.currentText()
        if not selected_isotope or selected_isotope not in self.isotope_dfs:
             self.filters_groupbox.setVisible(False)
             return

        isotope_df_for_filters = self.isotope_dfs[selected_isotope]

        # --- 1. Identify variables currently used in the plot ---
        plotted_vars = set()
        slider_var = None # Keep track of slider separately

        selected_plot_type = self.plot_type_combo.currentText()
        plot_type_code = 'unknown'; is_2d_heatmap = False
        if selected_plot_type == "3D Heatmap": plot_type_code = 'heatmap_scatter'
        elif selected_plot_type == "2D Heatmap": plot_type_code = 'heatmap_scatter'; is_2d_heatmap = True
        elif selected_plot_type == "3D Bubble Chart": plot_type_code = 'bubble'
        elif selected_plot_type == "3D Bar Chart": plot_type_code = 'bar3d'
        elif selected_plot_type == "Surface Plot": plot_type_code = 'surface'

        var1 = self.var_combo1.currentText()
        var2 = self.var_combo2.currentText()
        var3_raw = self.var_combo3.currentText()
        var4_raw = self.var_combo4.currentText()
        placeholders = ["No Variables Found", "Load CSV first", "No Element Variables Found", '', None]

        # Add roles based on plot type (same logic as in display_chart)
        x_axis_var = var1 if var1 not in placeholders else None
        y_axis_var, z_axis_var, color_var = None, None, None

        if plot_type_code == 'heatmap_scatter':
            if not is_2d_heatmap: # 3D
                y_axis_var = var2 if var2 not in placeholders else None
                z_axis_var = var3_raw if var3_raw not in placeholders else None
                slider_var = var4_raw if var4_raw not in placeholders else None
            else: # 2D
                 potential_sliders = [v for v in [var2, var4_raw] if v and v not in placeholders]
                 slider_var = potential_sliders[0] if potential_sliders else None
        elif plot_type_code == 'bubble':
            y_axis_var = var2 if var2 not in placeholders else None
            z_axis_var = var3_raw if var3_raw not in placeholders else None
            color_var = var4_raw if var4_raw not in placeholders else None
        elif plot_type_code == 'bar3d':
            y_axis_var = var2 if var2 not in placeholders else None
            color_var = var3_raw if var3_raw not in placeholders else None
            slider_var = var4_raw if var4_raw not in placeholders else None
        elif plot_type_code == 'surface':
            y_axis_var = var2 if var2 not in placeholders else None
            slider_var = var4_raw if var4_raw not in placeholders else None

        # Collect plotted/colored variables (excluding slider for now)
        if x_axis_var: plotted_vars.add(x_axis_var)
        if y_axis_var: plotted_vars.add(y_axis_var)
        if z_axis_var: plotted_vars.add(z_axis_var)
        if color_var: plotted_vars.add(color_var)

        # --- 2. Identify unused categorical variables ---
        all_categoricals_set = set(self.categorical_vars)
        unused_categoricals = all_categoricals_set - plotted_vars
        # Also remove the slider variable if it's categorical and unused otherwise
        if slider_var and slider_var in unused_categoricals:
            unused_categoricals.remove(slider_var)

        # --- 3. Create UI for each unused categorical variable ---
        if not unused_categoricals:
            self.filters_groupbox.setVisible(False) # Hide the group box if no filters needed
            return

        self.filters_groupbox.setVisible(True) # Show the group box
        print(f"Updating filters UI for unused categoricals: {sorted(list(unused_categoricals))}")

        for var_name in sorted(list(unused_categoricals)):
            # Double-check if the variable actually exists in the *current* isotope's DataFrame
            if var_name not in isotope_df_for_filters.columns:
                print(f"Warning: Unused categorical '{var_name}' not found in current isotope DF '{selected_isotope}'. Skipping filter UI.")
                continue

            filter_layout = QHBoxLayout()
            label = QLabel(f"{var_name}:")
            combo = QComboBox()
            # Add "All" first - it will be at index 0
            combo.addItem("All")

            try:
                # Get unique values, convert to string, sort
                unique_vals = isotope_df_for_filters[var_name].dropna().unique()
                # Ensure all values are strings before sorting
                sorted_vals = sorted([str(v) for v in unique_vals])
                # Add the actual category values - they will start at index 1
                combo.addItems(sorted_vals)

                # --- Set Default Selection ---
                # Check if any actual categories were added
                if combo.count() > 1:
                    # Set the current index to 1, which is the first actual category value (as 'all' is not recommended)
                    combo.setCurrentIndex(1)
                    print(f"  - Set default for '{var_name}' to index 1: '{combo.currentText()}'") # Debug print


            except Exception as e:
                print(f"Warning: Could not get unique values for filter '{var_name}': {e}")
                # Clear potentially partially populated items and set error state
                combo.clear()
                combo.addItem("All") # Still offer "All"
                combo.addItem("Error getting values")
                combo.setEnabled(False)
                # "All" will be the default in this error case

            filter_layout.addWidget(label)
            filter_layout.addWidget(combo)
            self.filters_layout.addLayout(filter_layout)
            self.unused_category_filters_widgets[var_name] = combo # Store combo for later access

    def display_chart(self):

        selected_isotope = self.isotope_combo.currentText() # BASE isotope name
        # --- Get user-selected plotting target ---
        plotting_target_col = self.target_variable_combo.currentText()
        # ---
        selected_plot_type = self.plot_type_combo.currentText()
        scale_type = self.scale_type_combo.currentText(); range_type = self.range_type_combo.currentText()
        fixed_min_val = None; fixed_max_val = None


        # --- Get raw selections from variable combos ---
        var1 = self.var_combo1.currentText()  # X
        var2 = self.var_combo2.currentText()  # Y or Slider
        var3_raw = self.var_combo3.currentText()  # Z / Color(Bar)
        var4_raw = self.var_combo4.currentText()  # Color(Bubble) / Slider


        if range_type == "Fixed Range":
            try: fixed_min_text = self.fixed_min_input.text().strip(); fixed_min_val = float(fixed_min_text) if fixed_min_text else None
            except ValueError: QMessageBox.warning(self, "Input Error", f"Invalid number for Fixed Min."); return
            try: fixed_max_text = self.fixed_max_input.text().strip(); fixed_max_val = float(fixed_max_text) if fixed_max_text else None
            except ValueError: QMessageBox.warning(self, "Input Error", f"Invalid number for Fixed Max."); return
            if fixed_min_val is not None and fixed_max_val is not None and fixed_min_val >= fixed_max_val: QMessageBox.warning(self, "Input Error", "Fixed Min must be less than Fixed Max."); return
            if fixed_min_val is None and fixed_max_val is None: QMessageBox.warning(self, "Input Error", "Enter values for Fixed Min and/or Fixed Max."); return
            if scale_type == "Logarithmic" and fixed_min_val is not None and fixed_min_val <= 0: QMessageBox.warning(self, "Input Error", "Fixed Min must be > 0 for Logarithmic scale."); return

        # ... get raw var selections, determine plot type code ...
        plot_type_code = 'unknown'; is_2d_heatmap = False
        if selected_plot_type == "3D Heatmap": plot_type_code = 'heatmap_scatter'
        elif selected_plot_type == "2D Heatmap": plot_type_code = 'heatmap_scatter'; is_2d_heatmap = True
        elif selected_plot_type == "3D Bubble Chart": plot_type_code = 'bubble'
        elif selected_plot_type == "3D Bar Chart": plot_type_code = 'bar3d'
        elif selected_plot_type == "Surface Plot": plot_type_code = 'surface'

        print(f"\n--- Preparing to Display Chart ---")
        print(f"Isotope Context: {selected_isotope}")
        print(f"Plotting Target Variable: {plotting_target_col}") # Log user choice
        print(f"Plot Type: {selected_plot_type} (Code: {plot_type_code}, 2D Heatmap: {is_2d_heatmap})")
        # --- Assign roles based on plot type ---
        x_axis_var, y_axis_var, z_axis_var, color_var, slider_var = None, None, None, None, None
        placeholders = ["No Variables Found", "Load CSV first", "No Element Variables Found", '', None]  # Define placeholders more robustly

        # Assign X axis (always Var1 if valid)
        x_axis_var = var1 if var1 not in placeholders else None

        # Assign other roles based on plot type
        if plot_type_code == 'heatmap_scatter':
            if not is_2d_heatmap:  # 3D Heatmap
                y_axis_var = var2 if var2 not in placeholders else None
                z_axis_var = var3_raw if var3_raw not in placeholders else None
                slider_var = var4_raw if var4_raw not in placeholders else None
            else:  # 2D Heatmap
                # Slider can be Var2 or Var4, check distinctness later
                potential_sliders = [v for v in [var2, var4_raw] if v and v not in placeholders]
                # Choose slider candidate (distinctness check happens later)
                slider_var = potential_sliders[0] if potential_sliders else None
                # y_axis_var, z_axis_var, color_var remain None
        elif plot_type_code == 'bubble':  # 3D Bubble
            y_axis_var = var2 if var2 not in placeholders else None
            z_axis_var = var3_raw if var3_raw not in placeholders else None
            color_var = var4_raw if var4_raw not in placeholders else None
            # No explicit slider from UI
        elif plot_type_code == 'bar3d':  # 3D Bar
            y_axis_var = var2 if var2 not in placeholders else None
            color_var = var3_raw if var3_raw not in placeholders else None  # Var3 is Color
            slider_var = var4_raw if var4_raw not in placeholders else None  # Var4 is Slider
        elif plot_type_code == 'surface':  # Surface Plot
            y_axis_var = var2 if var2 not in placeholders else None
            slider_var = var4_raw if var4_raw not in placeholders else None  # Allow slider filter via Var4

        print(
            f"Assigned Roles: X='{x_axis_var}', Y='{y_axis_var}', Z='{z_axis_var}', Color='{color_var}', Slider='{slider_var}'")

        # --- Basic Validation ---
        placeholders_isotope = ["Load CSV first", "No Isotopes Loaded", "No Isotopes Found", '', None]
        if not selected_isotope or selected_isotope in placeholders_isotope:
            QMessageBox.warning(self, "Selection Error", "Please select a valid isotope context.");
            return
        if not x_axis_var:  # X is always required
            QMessageBox.warning(self, "Selection Error", "Please select a valid variable for Axis 1 (X).");
            return

        # Collect variables required by the plot *elements*
        vars_for_plot_distinctness = {x_axis_var}
        if y_axis_var: vars_for_plot_distinctness.add(y_axis_var)
        if z_axis_var: vars_for_plot_distinctness.add(z_axis_var)
        if color_var: vars_for_plot_distinctness.add(color_var)
        # Remove None if present
        vars_for_plot_distinctness.discard(None)

        # --- Check for Distinct Variables based on Plot Type ---
        required_distinct_count = 0
        plot_desc = ""
        actual_vars_selected_plot = vars_for_plot_distinctness  # Already filtered None

        if plot_type_code == 'heatmap_scatter':
            if not is_2d_heatmap:  # 3D Heatmap (X, Y, Z)
                required_distinct_count = 3;
                plot_desc = "3D Heatmap (X, Y, Z)"
            else:  # 2D Heatmap (X)
                required_distinct_count = 1;
                plot_desc = "2D Heatmap (X)"
        elif plot_type_code == 'bubble':  # X, Y, Z, Color
            required_distinct_count = 4;
            plot_desc = "3D Bubble Chart (X, Y, Z, Color)"
        elif plot_type_code == 'bar3d':  # X, Y, Color
            required_distinct_count = 3;
            plot_desc = "3D Bar Chart (X, Y, Color)"
        elif plot_type_code == 'surface':  # X, Y
            required_distinct_count = 2;
            plot_desc = "Surface Plot (X, Y)"

        if len(actual_vars_selected_plot) < required_distinct_count:
            QMessageBox.warning(self, "Selection Error",
                                f"Please select {required_distinct_count} distinct, valid variables for the '{plot_desc}' plot elements.")
            return

        # --- Check slider variable distinctness & type ---
        if slider_var:  # Only check if a slider var was actually assigned
            if slider_var in actual_vars_selected_plot:
                QMessageBox.warning(self, "Selection Error",
                                    f"The slider variable ('{slider_var}') must be different from the variables used for plot elements.")
                return
            # Check against the identified categorical *element* variables
            if slider_var not in self.categorical_vars:
                QMessageBox.warning(self, "Selection Error",
                                    f"The slider variable ('{slider_var}') must be a categorical element variable (found in the 'Element Variables' dropdowns and detected as categorical).")
                return

        # --- Get Data (Check isotope context and user target column) ---
        if selected_isotope not in self.isotope_dfs:
             QMessageBox.critical(self, "Data Error", f"DataFrame for base isotope '{selected_isotope}' context not loaded or was skipped."); return
        original_isotope_data = self.isotope_dfs[selected_isotope].copy()

        # Validate the user-selected plotting target exists in the loaded DF
        placeholders_target = ["Load CSV first", "No Numeric Cols Found", '', None]
        if not plotting_target_col or plotting_target_col in placeholders_target:
             QMessageBox.warning(self, "Selection Error", "Please select a valid Target Variable for plotting."); return
        if plotting_target_col not in original_isotope_data.columns:
             QMessageBox.critical(self, "Data Error", f"Selected plotting target '{plotting_target_col}' not found in data for '{selected_isotope}'. It might be missing or non-numeric."); return
        print(f"Using plotting target column: '{plotting_target_col}'")

        # ---  Apply Filters for Unused Categorical Variables  ---
        filtered_df = original_isotope_data  # Start with the original data copy
        applied_explicit_filters = {}
        if self.filters_groupbox.isVisible() and self.unused_category_filters_widgets:
            print("Applying explicit filters for unused categorical variables...")
            for var_name, combo_box in self.unused_category_filters_widgets.items():
                selected_value = combo_box.currentText()
                if selected_value != "All" and selected_value != "Error getting values":
                    if var_name in filtered_df.columns:
                        try:
                            # Ensure comparison is robust (convert both sides to string)
                            filter_mask = filtered_df[var_name].astype(str) == str(selected_value)
                            filtered_df = filtered_df[filter_mask]
                            applied_explicit_filters[var_name] = selected_value
                            print(
                                f"  - Filtered by '{var_name}' = '{selected_value}'. Rows remaining: {len(filtered_df)}")
                            if filtered_df.empty:
                                QMessageBox.warning(self, "Filtering Warning",
                                                    f"No data remains after applying filter: '{var_name}' = '{selected_value}'.")
                                break  # Stop filtering if empty
                        except Exception as e:
                            QMessageBox.critical(self, "Filtering Error",
                                                 f"Error applying filter for '{var_name}' = '{selected_value}':\n{e}")
                            return  # Stop on filter error
                    else:
                        print(f"Warning: Filter variable '{var_name}' not found in DataFrame columns during filtering.")
            if applied_explicit_filters:
                print(f"Explicit filters applied: {applied_explicit_filters}")


        # Check if data remains after explicit filtering
        if filtered_df.empty and applied_explicit_filters:
            print("Plotting aborted: No data remaining after explicit filtering.")
            pass  # Continue to VI/ChartWindow launch

        # --- Calculate Variable Importance (IMPORTANT: using the filtered data - i.e. what unused variables you have chosen) ---
        print(f"Preparing data for Random Forest Variable Importance on potentially filtered data...")
        rf_target_col = plotting_target_col
        self.variable_importance = pd.DataFrame(columns=['Feature', 'Importance'])  # Reset VI

        # --- Use 'filtered_df' for VI calculation ---
        rf_prep_df = filtered_df  # Use the explicitly filtered data

        if not rf_target_col or rf_target_col in placeholders_target:
            print(f"Warning: Invalid target variable selected ('{rf_target_col}'). Skipping VI.")
        elif rf_target_col not in rf_prep_df.columns:
            print(
                f"Warning: Selected target '{rf_target_col}' not found in (filtered) data for '{selected_isotope}'. Skipping VI.")
        else:
            # Features are categorical element vars present in the filtered data
            X_columns_rf = [col for col in self.categorical_vars if col in rf_prep_df.columns]
            if rf_target_col in X_columns_rf: X_columns_rf.remove(rf_target_col)

            if not X_columns_rf:
                print(
                    "Warning: No suitable categorical element variables found in (filtered) data for RF. Skipping VI.")
            else:
                y_rf = pd.to_numeric(rf_prep_df[rf_target_col], errors='coerce')
                if y_rf.isnull().all():
                    print(f"Warning: Target '{rf_target_col}' is all NaN in filtered data. Skipping VI.")
                elif y_rf.nunique() <= 1:
                    print(f"Warning: Target '{rf_target_col}' has <= 1 unique value in filtered data. Skipping VI.")
                else:
                    X_df_rf = rf_prep_df[X_columns_rf].copy()
                    for col in X_columns_rf:  # Fill NaNs before encoding
                        if X_df_rf[col].isnull().any():
                            if not pd.api.types.is_string_dtype(X_df_rf[col]): X_df_rf[col] = X_df_rf[col].astype(
                                str)
                            X_df_rf[col] = X_df_rf[col].fillna('Missing_VI')

                    print(f"RF Features (on filtered data): {X_columns_rf}")
                    try:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                        valid_target_indices = y_rf.notna()
                        if not valid_target_indices.any():
                            print("Warning: No valid target values found in filtered data. Skipping RF.")
                        else:
                            X_to_encode = X_df_rf.loc[valid_target_indices]
                            y_train_ready = y_rf.loc[valid_target_indices]
                            X_to_encode = X_to_encode.dropna()  # Drop rows with NaNs in features (should be handled)
                            y_train_ready = y_train_ready.loc[X_to_encode.index]

                            if X_to_encode.empty:
                                print("Warning: No rows remained for encoding after filtering/dropna. Skipping VI.")
                            else:
                                encoded_data = encoder.fit_transform(X_to_encode)
                                if encoded_data.shape[1] == 0:
                                    print("Warning: No features after OHE on filtered data. Skipping VI.")
                                else:
                                    # ... (rest of RF training logic - KEEP AS IS) ...
                                    encoded_feature_names = encoder.get_feature_names_out(X_columns_rf)
                                    X_train_ready = pd.DataFrame(encoded_data, columns=encoded_feature_names,
                                                                 index=X_to_encode.index)
                                    if len(X_train_ready) < 10:
                                        print(
                                            f"Warning: Too few samples ({len(X_train_ready)}) for RF after filtering. Skipping VI.")
                                    else:
                                        try:
                                            X_train, X_test, y_train, y_test = train_test_split(X_train_ready,
                                                                                                y_train_ready,
                                                                                                test_size=0.2,
                                                                                                random_state=42)
                                            print(f"Training RF model on {len(X_train)} filtered samples...")
                                            rf_model = RandomForestRegressor(n_estimators=50, random_state=42,
                                                                             n_jobs=-1, max_depth=10,
                                                                             min_samples_leaf=5, oob_score=False)
                                            rf_model.fit(X_train, y_train)
                                            importances = rf_model.feature_importances_
                                            self.variable_importance = pd.DataFrame(
                                                {'Feature': X_train_ready.columns,
                                                 'Importance': importances}).sort_values(by='Importance',
                                                                                         ascending=False).reset_index(
                                                drop=True)
                                            print(
                                                f"VarImp (on filtered data) relative to '{rf_target_col}' (Top 5):\n",
                                                self.variable_importance.head())
                                        except Exception as e_rf:
                                            print(
                                                f"Error during RF fitting on filtered data: {e_rf}"); traceback.print_exc()

                    except Exception as e_enc:
                        print(f"Error during encoding/prep on filtered data: {e_enc}"); traceback.print_exc()

        # --- Launch Chart Window (passing the potentially filtered DF) ---
        if self.chart_window and self.chart_window.isVisible(): self.chart_window.close()
        self.chart_window = None
        try:
            print("Creating and showing ChartWindow with potentially filtered data...");
            initial_slider_index = 0  # Slider index logic remains the same

            # --- Pass 'filtered_df' to ChartWindow ---
            self.chart_window = ChartWindow(
                selected_isotope_prefix=selected_isotope,
                plot_type=plot_type_code,
                var_axis1=x_axis_var, var_axis2=y_axis_var, var_axis3_or_mode=z_axis_var,
                var_color=color_var, slider_var=slider_var,
                initial_slider_value=initial_slider_index,
                importance_df=self.variable_importance,  # VI based on filtered data
                isotope_df=filtered_df,  # Pass the filtered data here
                value_col=plotting_target_col,
                categorical_vars_list=self.categorical_vars,  # Pass full list for potential encoding
                continuous_vars_list=self.continuous_vars,  # Pass full list
                scale_type=scale_type, range_type=range_type,
                fixed_min=fixed_min_val, fixed_max=fixed_max_val
            )
            self.chart_window.show()
            print("ChartWindow launched successfully.")
        except Exception as e:
            error_message = f"Failed to create or show Chart Window:\n{e}";
            print(error_message);
            traceback.print_exc();
            QMessageBox.critical(self, "Chart Window Error", error_message)

# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())