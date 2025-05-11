from masking import *
from jpformer import JPFormer
import argparse
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import sys
import time
import json
import torch
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import threading
import queue

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, "models", "jpformer"))
sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))


class GlucoseDashboard:
    def __init__(self, root, ptid=None, optimised_for="hypo"):
        self.root = root
        self.root.title("Blood Glucose Monitor")

        # default size
        self.root.geometry("1800x1100")

        # theme and appearance
        self.root.configure(bg="#f0f0f0")

        # custom colors
        self.HYPO_COLOR = (173 / 255, 29 / 255, 30 / 255)
        self.NORMAL_COLOR = (98 / 255, 145 / 255, 117 / 255)
        self.HYPER_COLOR = (248 / 255, 151 / 255, 33 / 255)
        self.GLUCOSE_LINE_COLOR = (80 / 255, 80 / 255, 80 / 255)

        # convert RGB tuples to hex for Tkinter
        self.HYPO_HEX = "#{:02x}{:02x}{:02x}".format(
            int(self.HYPO_COLOR[0] * 255),
            int(self.HYPO_COLOR[1] * 255),
            int(self.HYPO_COLOR[2] * 255)
        )
        self.NORMAL_HEX = "#{:02x}{:02x}{:02x}".format(
            int(self.NORMAL_COLOR[0] * 255),
            int(self.NORMAL_COLOR[1] * 255),
            int(self.NORMAL_COLOR[2] * 255)
        )
        self.HYPER_HEX = "#{:02x}{:02x}{:02x}".format(
            int(self.HYPER_COLOR[0] * 255),
            int(self.HYPER_COLOR[1] * 255),
            int(self.HYPER_COLOR[2] * 255)
        )

        # model parameters
        self.ptid = ptid
        self.optimised_for = optimised_for
        self.model = None
        self.device = None

        # initialize variables
        self.current_glucose = None
        self.predictions_df = None
        self.hypo_warning = None
        self.hyper_warning = None
        self.last_timestamp = None
        self.df = None
        self.starting_index = 72

        # for handling model inference in background
        self.data_queue = queue.Queue()
        self.running = True

        # thread synchronisation - prevent inference before model is ready
        self.model_initialised = threading.Event()

        # set up the interface
        self.setup_ui()

        # initialise model and inference loops in separate threads to avoid ordering error
        self.model_thread = threading.Thread(
            target=self.initialize_model_and_data)
        self.model_thread.daemon = True
        self.model_thread.start()

        self.inference_thread = threading.Thread(
            target=self.inference_loop_thread)
        self.inference_thread.daemon = True
        self.inference_thread.start()

        #  UI update schedule
        self.root.after(100, self.check_queue)

    def setup_ui(self):
        """Create user interface"""

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        #  patient ID and model info
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text=f"Patient ID: {self.ptid}", font=(
            "montserrat", 14, "bold")).pack(side=tk.LEFT)
        self.model_info_label = ttk.Label(
            header_frame, text="Loading model information...", font=("montserrat", 12))
        self.model_info_label.pack(side=tk.RIGHT)

        # frame for current glucose and warnings
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 20))

        # current glucose frame
        glucose_frame = ttk.Frame(
            top_frame, padding=10, relief="ridge", borderwidth=2)
        glucose_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(glucose_frame, text="Current Blood Glucose",
                  font=("montserrat", 24, "bold")).pack(anchor="w")

        # glucose value and status frames
        glucose_value_frame = ttk.Frame(glucose_frame, height=80)
        glucose_value_frame.pack(fill=tk.X)
        glucose_value_frame.pack_propagate(False)  # Prevent automatic resizing

        self.glucose_value = ttk.Label(
            glucose_value_frame, text="-- mg/dL", font=("montserrat", 36, "bold"))
        self.glucose_value.pack(anchor="w", pady=5)

        self.glucose_status = ttk.Label(
            glucose_value_frame, text="", font=("montserrat", 24))
        self.glucose_status.pack(anchor="w")

        self.last_updated = ttk.Label(
            glucose_frame, text="Last updated: --", font=("montserrat", 16))
        self.last_updated.pack(anchor="w", pady=5)

        # warnings frame
        warnings_frame = ttk.Frame(
            top_frame, padding=10, relief="ridge", borderwidth=2)
        warnings_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(warnings_frame, text="Warnings", font=(
            "montserrat", 24, "bold")).pack(anchor="w")

        # inner frame for warnings - warning text
        warning_frame = ttk.Frame(
            warnings_frame, height=100)  # Taller single frame
        warning_frame.pack(fill=tk.X, pady=15)
        # Prevent the frame from resizing based on content
        warning_frame.pack_propagate(False)

        self.warning_label = ttk.Label(
            warning_frame, text="No warnings", font=("montserrat", 24))
        # Center align with padding for vertical center
        self.warning_label.pack(anchor="w", pady=5)

        # Chart frame
        chart_frame = ttk.Frame(main_frame, padding=10,
                                relief="ridge", borderwidth=2)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(chart_frame, text="Predicted Blood Glucose (Next 2 Hours)",
                  font=("montserrat", 24, "bold")).pack(anchor="w", pady=(0, 10))

        # container for matplotlib chart
        chart_container = ttk.Frame(chart_frame, height=300)  # Fixed height
        chart_container.pack(fill=tk.BOTH, expand=True)
        # Prevent resizing based on content
        chart_container.pack_propagate(False)

        self.fig = plt.figure(figsize=(10, 5), dpi=100)

        self.ax = self.fig.add_axes(
            [0.1, 0.2, 0.85, 0.70])  # Fixed axes position

        # matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_plot()

        # footer
        footer_frame = ttk.Frame(main_frame, padding=10)
        footer_frame.pack(fill=tk.X, pady=(20, 0))

        self.status_label = ttk.Label(footer_frame, text="Initializing model and data...",
                                      foreground="white")
        self.status_label.pack(anchor="w")

    def setup_plot(self):
        """Configure the matplotlib plot"""

        # disable auto layout to prevent plot resizing
        plt.rcParams.update({'figure.autolayout': False})

        # set axis limits
        self.ax.set_xlabel("Time", fontsize=8)
        self.ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=8)
        self.ax.set_ylim(0, 400)  # Updated y-axis range

        self.ax.tick_params(axis='both', which='major', labelsize=7)

        # reference lines
        self.ax.axhline(y=70, color=self.HYPO_COLOR,
                        linestyle='--', alpha=0.7, label='Hypo', linewidth=1)
        self.ax.axhline(y=80, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=130, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=180, color=self.HYPER_COLOR,
                        linestyle='--', alpha=0.7, label='Hyper', linewidth=1)

        self.ax.legend(loc='upper right', fontsize=7)

        self.ax.grid(False)

        self.line, = self.ax.plot(
            [], [], '-o', color=self.GLUCOSE_LINE_COLOR, linewidth=1.0, markersize=2)

    def initialize_model_and_data(self):
        """Initialize the JPFormer model and load data"""
        try:
            # model directories for population and personalised models
            population_model_dir = os.path.join(
                PROJECT_ROOT, 'models/jpformer/population_jpformer_final_model/population_jpformer_ohio_ptid_results')
            personalised_model_dir = os.path.join(
                PROJECT_ROOT, 'models/jpformer/fine_tuning_development_files/loss_function_weights_lowest')

            # load saved results for population and personalised models
            # used to determine best performing model based on the optimised_for parameter
            try:
                population_performance_df = pd.read_csv(os.path.join(
                    population_model_dir, f"patient_{self.ptid}/base_model_eval/patient_{self.ptid}_base_model_overall_cg_ega.csv"))
                personalised_performance_df = pd.read_csv(os.path.join(
                    personalised_model_dir, f"patient_{self.ptid}/fine_tuning_eval/patient_{self.ptid}_overall_cg_ega.csv"))

                population_performance_df['EP%'] = (
                    population_performance_df['EP'] / population_performance_df['Count']) * 100
                personalised_performance_df['EP%'] = (
                    personalised_performance_df['EP'] / personalised_performance_df['Count']) * 100

                if self.optimised_for == 'hypo':
                    population_percent = population_performance_df['EP%'].iloc[0]
                    personalised_percent = personalised_performance_df['EP%'].iloc[0]
                else:
                    population_percent = population_performance_df['EP%'].iloc[-1]
                    personalised_percent = personalised_performance_df['EP%'].iloc[-1]

                use_personalised = population_percent > personalised_percent
            except Exception as e:
                print(f"Error loading performance data: {str(e)}")
                use_personalised = False  # Default to population model if error

            # set the model directory and config path based on the best performing model
            if use_personalised:
                model_name = "Personalised Model"
                personalised_model_dir = os.path.join(
                    personalised_model_dir, f"patient_{self.ptid}/fine_tuning_eval")
                best_personalised_model_file = [f for f in os.listdir(
                    personalised_model_dir) if f.endswith('.pth')]

                if not best_personalised_model_file:
                    raise FileNotFoundError("No personalised model file found")

                pretrained_weights_path = os.path.join(
                    personalised_model_dir, best_personalised_model_file[0])
                config_path = os.path.join(
                    PROJECT_ROOT, "models/shared_config_files/fine_tuning_config.json")
            else:
                model_name = "Population Model"
                pretrained_weights_path = os.path.join(
                    PROJECT_ROOT,
                    "models",
                    "jpformer",
                    "population_jpformer_final_model",
                    "population_jpformer_replace_bg_aggregate_results",
                    "jpformer_dual_weighted_rmse_loss_func_high_dim_4_enc_lyrs_high_dropout_0.5696_MAE_0.3965.pth"
                )
                config_path = os.path.join(
                    PROJECT_ROOT, "models/shared_config_files/final_models_config.json")

            # load config and set up device
            config = self.load_config(config_path)
            config = self.ConfigObject(config)
            self.device = self.setup_device(config)

            # load model
            self.model, model_class_name = self.load_model(
                config, self.device, pretrained_weights_path)

            # load  patient data
            self.df = self.get_full_ohio_ptid_data(self.ptid)

            # update UI with model info
            optimised_for_text = (
                "minimising hypoglycaemic errors"
                if self.optimised_for == "hypo"
                else "minimising overall prediction error"
            )
            self.data_queue.put(
                {"model_info": f"Using {model_name}: {model_class_name} ({optimised_for_text})"})
            self.model_info_label.config(font=("montserrat", 14))
            self.data_queue.put(
                {"status": f"Loaded data for PtID {self.ptid} with {len(self.df)} records"})

            self.model_initialised.set()

        except Exception as e:
            print(f"Error in model initialisation: {str(e)}")
            self.data_queue.put({"model_info": "Error loading model"})
            self.data_queue.put({"status": f"Error: {str(e)}"})

    def inference_loop_thread(self):
        """Thread function for continuous inference"""
        # wait for model initialisation to complete before starting inference to
        # avoid ordering errors
        if self.model_initialised.wait(timeout=60):
            print("Model initialisation complete, starting inference loop")
            self.model_inference_loop()
        else:
            print("Model initialisation timed out")
            self.data_queue.put(
                {"status": "ERROR: Model initialisation timed out. Please restart the application."})

    def update_timestamp(self, timestamp):
        """Update only the timestamp display"""

        if not isinstance(timestamp, datetime):
            timestamp = pd.to_datetime(timestamp)

        formatted_time = timestamp.strftime('%H:%M:%S')

        self.last_updated.config(
            text=f"Time: {formatted_time}",
            foreground="white",
            font=("montserrat", 18, "bold")
        )

    def update_current_glucose(self, current_glucose):
        """Update only the current glucose display"""
        # skip update if glucose value is None
        if current_glucose is None:
            self.glucose_value.config(text="-- mg/dL", foreground="black")
            self.glucose_status.config(
                text="Status: Unknown",
                foreground="black",
                font=("montserrat", 18, "bold")
            )
            return

        # update current glucose display
        status, color = self.get_glucose_status(current_glucose)

        # show current glucose value and status
        self.glucose_value.config(
            text=f"{current_glucose:.1f} mg/dL", foreground=color)

    def clear_warnings(self, current_glucose):
        """
        Clear prediction warnings but ALWAYS show warnings for current glucose values if needed
        """
        # if current glucose in warning range, show warnings
        if current_glucose is not None:
            if current_glucose < 70:
                self.warning_label.config(
                    text="WARNING: Current Hypoglycaemia - Consider Action!",
                    foreground=self.HYPO_HEX,
                    font=("montserrat", 24, "bold")
                )
                return
            elif current_glucose > 180:
                self.warning_label.config(
                    text="WARNING: Current Hyperglycaemia - Consider Action!",
                    foreground=self.HYPER_HEX,
                    font=("montserrat", 24, "bold")
                )
                return
            else:
                # show "No predictions available" if current glucose is normal
                self.warning_label.config(
                    text="No predictions available",
                    foreground=self.HYPO_HEX,
                    # Use normal weight instead of bold
                    font=("montserrat", 24, "normal")
                )
        else:
            self.warning_label.config(
                text="No data available",
                foreground=self.HYPO_HEX,
                font=("montserrat", 24, "normal")
            )

    def show_missing_data_message(self, current_timestamp=None):
        """Display a message when consecutive missing values are detected."""
        # Clear the chart but maintain settings
        self.ax.clear()

        self.ax.set_xlabel("Time", fontsize=8, labelpad=10, fontweight="bold")
        self.ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=8,
                           labelpad=10, fontweight="bold")
        self.ax.set_ylim(0, 400)

        self.ax.axhline(y=70, color=self.HYPO_COLOR,
                        linestyle='--', alpha=0.7, label='Hypo', linewidth=1)
        self.ax.axhline(y=80, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=130, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=180, color=self.HYPER_COLOR,
                        linestyle='--', alpha=0.7, label='Hyper', linewidth=1)

        # dont show x ticks as there are no predictions
        self.ax.set_xticks([])

        self.ax.grid(False)

        # display missing data message
        self.ax.text(0.5, 0.7, 'No predictions available\n\nConsecutive missing values detected',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=self.ax.transAxes,
                     fontsize=12)

        self.canvas.draw()

    def update_predictions(self, predictions_df, hypo_warning, hyper_warning):
        """Update prediction warnings and chart"""

        # update timestamp if available in predictions_df
        if predictions_df is not None and len(
                predictions_df) > 0 and 'timestamp' in predictions_df.columns:
            # first prediction is 5 minutes in the future, current time is 5 min before that
            current_time = predictions_df['timestamp'].iloc[0] - \
                pd.Timedelta(minutes=5)
            self.update_timestamp(current_time)

        # check if the current glucose is in a warning range
        if hasattr(self, 'current_glucose') and self.current_glucose is not None:
            current_hypo = self.current_glucose < 70
            current_hyper = self.current_glucose > 180
        else:
            current_hypo = False
            current_hyper = False

        # update warnings based on current glucose and prediction results
        if current_hypo:
            self.warning_label.config(
                text="WARNING: Current Hypoglycaemia - Consider Action!",
                foreground=self.HYPO_HEX,
                font=("montserrat", 24, "bold")
            )
        elif current_hyper:
            self.warning_label.config(
                text="WARNING: Current Hyperglycaemia - Consider Action!",
                foreground=self.HYPER_HEX,
                font=("montserrat", 24, "bold")
            )

        elif not hypo_warning and not hyper_warning:
            self.warning_label.config(
                text="Blood Glucose is Stable",
                foreground=self.NORMAL_HEX,
                font=("montserrat", 24, "bold")
            )

        elif hypo_warning and hyper_warning:
            hypo_time = hypo_warning.strftime("%H:%M") if isinstance(
                hypo_warning, datetime) else hypo_warning
            hyper_time = hyper_warning.strftime("%H:%M") if isinstance(
                hyper_warning, datetime) else hyper_warning
            if hypo_time < hyper_time:
                self.warning_label.config(
                    text=(
                        f"WARNING: Hypoglycaemia Predicted at {hypo_time}\n"
                        f"Hyperglycaemia Predicted at {hyper_time}"
                    ),
                    foreground=self.HYPO_HEX,
                    font=("montserrat", 24, "bold")
                )
            else:
                self.warning_label.config(
                    text=(
                        f"WARNING: Hyperglycaemia Predicted at {hyper_time}\n"
                        f"Hypoglycaemia Predicted at {hypo_time}"
                    ),
                    foreground=self.HYPER_HEX,
                    font=("montserrat", 24, "bold")
                )

        elif hypo_warning:
            hypo_time = hypo_warning.strftime("%H:%M") if isinstance(
                hypo_warning, datetime) else hypo_warning
            self.warning_label.config(
                text=f"WARNING: Hypoglycaemia Predicted at {hypo_time}",
                foreground=self.HYPO_HEX,
                font=("montserrat", 24, "bold")
            )

        elif hyper_warning:
            hyper_time = hyper_warning.strftime("%H:%M") if isinstance(
                hyper_warning, datetime) else hyper_warning
            self.warning_label.config(
                text=f"WARNING: Hyperglycaemia Predicted at {hyper_time}",
                foreground=self.HYPER_HEX,
                font=("montserrat", 24, "bold")
            )

        self.update_chart(predictions_df)

    def update_chart(self, predictions_df):
        """Update the prediction chart with new data"""

        self.ax.clear()

        if predictions_df is None or len(predictions_df) == 0:
            self.show_missing_data_message()
            return

        # times for the x-axis
        if 'timestamp' in predictions_df.columns:
            # Convert all timestamps to datetime objects if they aren't already
            if not isinstance(predictions_df['timestamp'].iloc[0], datetime):
                try:
                    predictions_df['timestamp'] = pd.to_datetime(
                        predictions_df['timestamp'])
                except Exception as e:
                    print(f"Error converting timestamps: {str(e)}")
                    formatted_times = [
                        f"T+{i * 5}" for i in range(len(predictions_df))]
            else:
                # format times as HH:MM
                formatted_times = [t.strftime("%H:%M")
                                   for t in predictions_df['timestamp']]
        else:
            # if there's no timestamp, just use sequence numbers
            formatted_times = [f"T+{i * 5}" for i in range(len(predictions_df))]

        x_positions = list(range(len(formatted_times)))

        # redraw reference lines
        self.ax.axhline(y=70, color=self.HYPO_COLOR,
                        linestyle='--', alpha=0.7, label='Hypo', linewidth=1)
        self.ax.axhline(y=80, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=130, color=self.NORMAL_COLOR, linestyle='--',
                        alpha=0.7, label='Target Range', linewidth=0.7)
        self.ax.axhline(y=180, color=self.HYPER_COLOR,
                        linestyle='--', alpha=0.7, label='Hyper', linewidth=1)

        # plot new data using numeric x positions
        glucose_values = predictions_df['glucose_value'].values
        self.ax.plot(x_positions, glucose_values, '-o',
                     color=self.GLUCOSE_LINE_COLOR, linewidth=1.0, markersize=3)

        # highlight if predictions cross thresholds with custom colors
        for i, value in enumerate(glucose_values):
            if value < 70:
                self.ax.scatter(
                    x_positions[i], value, color=self.HYPO_COLOR, s=50, alpha=0.5)
            elif value > 180:
                self.ax.scatter(
                    x_positions[i], value, color=self.HYPER_COLOR, s=50, alpha=0.5)

        self.ax.set_xlabel("Time", fontsize=6, fontweight="bold", labelpad=15)
        self.ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=6,
                           fontweight="bold", labelpad=15)
        self.ax.set_ylim(0, 400)
        self.ax.tick_params(axis='both', which='major', labelsize=7)

        # Every third position
        tick_positions = list(range(0, len(formatted_times), 3))
        tick_labels = [formatted_times[i] for i in tick_positions]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)

        self.ax.grid(False)

        # Set up legend to only show target range once
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        self.ax.legend(unique_handles, unique_labels,
                       loc='upper right', fontsize=6, frameon=False)

        self.canvas.draw()

    def update_dashboard(self, current_glucose, predictions_df,
                         hypo_warning, hyper_warning, last_timestamp=None):
        """Update all dashboard elements with new data (keeping for backward compatibility)"""
        # Update timestamp
        if last_timestamp is not None:
            self.update_timestamp(last_timestamp)

        # Update current glucose
        self.update_current_glucose(current_glucose)

        # Update predictions
        if predictions_df is not None:
            self.update_predictions(
                predictions_df, hypo_warning, hyper_warning)
        else:
            self.clear_warnings(current_glucose)

    def check_queue(self):
        """Check for data updates and update the UI"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()

                # update model
                if "model_info" in data:
                    self.model_info_label.config(text=data["model_info"])

                if "status" in data:
                    self.status_label.config(text=data["status"])

                # always update timestamp
                if "last_timestamp" in data and data["last_timestamp"] is not None:
                    self.update_timestamp(data["last_timestamp"])

                # update current glucose, regardless of missing data flag
                if "current_glucose" in data:
                    self.update_current_glucose(data["current_glucose"])

                # handle missing data cases for predictions
                if "missing_data" in data and data["missing_data"]:

                    self.show_missing_data_message()

                    # prioritise current glucose warnings over "no predictions" message
                    if "current_glucose" in data:
                        self.clear_warnings(data["current_glucose"])
                    else:
                        self.clear_warnings(None)

                # only update prediction data if predictions are available
                elif "predictions_df" in data:
                    self.update_predictions(
                        data["predictions_df"],
                        data["hypo_warning"],
                        data["hyper_warning"]
                    )

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def ohio_data_slicing(self, df, start_index):
        """Create sliding windows of data for glucose prediction.
        Returns current glucose value even if prediction isn't possible."""
        input_slice = df[start_index:start_index + 72].reset_index(drop=True)

        # extract current glucose data - last row of the input slice
        if len(input_slice) >= 72:
            current_glucose_normalised = input_slice['glucose_value'].iloc[-1]
            current_timestamp = input_slice['timestamp'].iloc[-1]

            # denormalize current glucose value
            mean_bg = 152.91051040286524
            std_bg = 70.27050122812615
            current_glucose = (current_glucose_normalised * std_bg) + mean_bg
        else:
            current_glucose = None
            current_timestamp = None
            return None, None, None, current_glucose, current_timestamp

        if len(input_slice) < 72:
            print(f"Not enough data points: got {len(input_slice)}, need 72")
            return None, None, None, current_glucose, current_timestamp

        if input_slice['RollingTimeDiffFlag'].iloc[-1] != 72:
            print(
                "Unable to predict accurate BG values as input data contains "
                "consecutive missing values"
            )
            return None, None, None, current_glucose, current_timestamp

        input_slice = input_slice.drop(columns=['RollingTimeDiffFlag'])

        encoder_input = input_slice.iloc[:72]
        if 'timestamp' in encoder_input.columns:
            encoder_input = encoder_input.drop(columns=['timestamp'])

        start_token = input_slice.iloc[-12:]
        last_timestamp = start_token['timestamp'].iloc[-1]
        start_token = start_token.drop(columns=['timestamp'])

        decoder_time_sequence = pd.DataFrame({
            'glucose_value': [0] * 24,
            'timestamp': pd.date_range(
                start=last_timestamp + pd.Timedelta(minutes=5),
                periods=24,
                freq='5min'
            ),
        })

        decoder_time_sequence['hour'] = decoder_time_sequence['timestamp'].dt.hour
        decoder_time_sequence['minute'] = decoder_time_sequence['timestamp'].dt.minute
        decoder_time_sequence = decoder_time_sequence.drop(columns=[
                                                           'timestamp'])

        decoder_input = pd.concat(
            [start_token, decoder_time_sequence], ignore_index=True)

        encoder_input = torch.tensor(encoder_input.values, dtype=torch.float32)
        decoder_input = torch.tensor(decoder_input.values, dtype=torch.float32)

        return encoder_input, decoder_input, last_timestamp, current_glucose, current_timestamp

    def model_inference_loop(self):
        """Actual model inference"""
        while self.running:
            try:
                # ensure valid dataframe
                if self.df is None:
                    print("Warning: DataFrame is Empty. Waiting...")
                    continue

                # keep in range of dataframe
                if self.starting_index >= len(self.df) - 24:
                    # Reached the end of data, loop back
                    self.data_queue.put({"status": "Reached end of data"})
                    break

                # get input data, current glucose value, and timestamps from the data
                # slicing function
                encoder_input, decoder_input, last_timestamp, current_glucose, \
                    current_timestamp = self.ohio_data_slicing(
                        self.df, self.starting_index
                    )

                can_predict = (
                    encoder_input is not None
                    and decoder_input is not None
                    and last_timestamp is not None
                )

                if not can_predict:
                    #  could still have current glucose that should be displayed
                    self.data_queue.put({
                        "status": (
                            f"Skipping index {self.starting_index} - "
                            "Missing data detected or consecutive missing values"
                        ),
                        "missing_data": True,
                        "current_glucose": current_glucose,
                        "last_timestamp": current_timestamp
                    })
                    self.starting_index += 1
                    time.sleep(1)
                    continue

                # inference
                hypo_time, hyper_time, output_df = self.inference_loop(
                    encoder_input, decoder_input, last_timestamp, self.model, self.device)

                # update dashboard with new data
                self.data_queue.put({
                    "current_glucose": current_glucose,
                    "predictions_df": output_df,
                    "hypo_warning": hypo_time,
                    "hyper_warning": hyper_time,
                    "last_timestamp": last_timestamp,
                    "status": f"Processing data point {self.starting_index} of {len(self.df)}",
                    "missing_data": False  # Flag to indicate valid data
                })

                # sleep for 1.5 seconds to simulate real-time processing
                time.sleep(1.5)
                self.starting_index += 1

            except Exception as e:
                print(f"Error during inference: {str(e)}")
                self.data_queue.put({
                    "status": f"Error during inference: {str(e)}",
                    "missing_data": True  # Flag errors as missing data too
                })

    def get_full_ohio_ptid_data(self, ptid):
        """Read and Process Ohio Patient Data"""
        ptid_file = os.path.join(PROJECT_ROOT, 'data', 'source_data',
                                 'SourceData', 'Ohio', 'Test', f"{ptid}-ws-testing.xml")

        # parse the XML file
        tree = ET.parse(ptid_file)
        root = tree.getroot()
        data = []

        # extract Patient ID (as an integer)
        file_ptid = int(root.attrib['id'])

        assert file_ptid == ptid, (
            f"Patient ID in file {file_ptid} does not match provided PtID {ptid}"
        )

        # Eextract glucose level events including timestamp and value
        for event in root.find('glucose_level').findall('event'):
            row = {'timestamp': event.attrib['ts'],
                   'glucose_value': event.attrib['value']}
            data.append(row)

        # create a DataFrame for the patient
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        df = df.sort_values(by='timestamp', ascending=True)

        df['real_value_flag'] = 1
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

        mask = (df['time_diff'] > 595) & (df['time_diff'] < 605)
        insert_rows = df[mask].copy()

        if not insert_rows.empty:
            # modify new rows
            insert_rows['real_value_flag'] = 0
            insert_rows['timestamp'] -= pd.to_timedelta(5, unit='m')
            insert_rows['glucose_value'] = np.nan

        df = pd.concat([df, insert_rows],
                       ignore_index=True).reset_index(drop=True)

        df['glucose_value'] = df['glucose_value'].astype(float)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute

        df['time_diff_flag'] = df['time_diff'].apply(
            lambda x: 0 if x < 295 or x > 305 else 1)
        df['RollingTimeDiffFlag'] = df['time_diff_flag'].rolling(
            window=72).sum()

        df = df.drop(
            columns=['time_diff', 'time_diff_flag', 'real_value_flag'])

        # bg_value z-score normalisation
        df['glucose_value'] = (df['glucose_value'] -
                               152.91051040286524) / 70.27050122812615

        return df

    def ohio_data_slicing(self, df, start_index):
        """Create sliding windows of data for glucose prediction.
        Returns current glucose value even if prediction isn't possible."""
        input_slice = df[start_index:start_index + 72].reset_index(drop=True)

        # extract current glucose data regardless of prediction capability
        # the current glucose is always the last row of the input slice
        if len(input_slice) >= 72:
            current_glucose_normalised = input_slice['glucose_value'].iloc[-1]
            current_timestamp = input_slice['timestamp'].iloc[-1]

            # denormalize the current glucose value
            mean_bg = 152.91051040286524
            std_bg = 70.27050122812615
            current_glucose = (current_glucose_normalised * std_bg) + mean_bg
        else:
            current_glucose = None
            current_timestamp = None
            return None, None, None, current_glucose, current_timestamp

        if len(input_slice) < 72:
            print(f"Not enough data points: got {len(input_slice)}, need 72")
            return None, None, None, current_glucose, current_timestamp

        if input_slice['RollingTimeDiffFlag'].iloc[-1] != 72:
            return None, None, None, current_glucose, current_timestamp

        input_slice = input_slice.drop(columns=['RollingTimeDiffFlag'])

        encoder_input = input_slice.iloc[:72]
        if 'timestamp' in encoder_input.columns:
            encoder_input = encoder_input.drop(columns=['timestamp'])

        start_token = input_slice.iloc[-12:]
        last_timestamp = start_token['timestamp'].iloc[-1]
        start_token = start_token.drop(columns=['timestamp'])

        decoder_time_sequence = pd.DataFrame({
            'glucose_value': [0] * 24,
            'timestamp': pd.date_range(
                start=last_timestamp + pd.Timedelta(minutes=5),
                periods=24,
                freq='5min'
            ),
        })

        decoder_time_sequence['hour'] = decoder_time_sequence['timestamp'].dt.hour
        decoder_time_sequence['minute'] = decoder_time_sequence['timestamp'].dt.minute
        decoder_time_sequence = decoder_time_sequence.drop(columns=[
                                                           'timestamp'])

        decoder_input = pd.concat(
            [start_token, decoder_time_sequence], ignore_index=True)

        # convert to tensors
        encoder_input = torch.tensor(encoder_input.values, dtype=torch.float32)
        decoder_input = torch.tensor(decoder_input.values, dtype=torch.float32)

        return encoder_input, decoder_input, last_timestamp, current_glucose, current_timestamp

    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as file:
            return json.load(file)

    class ConfigObject:
        """Convert a dictionary to an object with attributes."""

        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    def setup_device(self, config):
        """Set up and return the appropriate computation device based on availability."""
        if torch.cuda.is_available() and config.use_gpu:
            device = torch.device(f"cuda:{config.gpu}")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon (M1/M2)
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def load_model(self, config, device, pretrained_weights_path=None):
        """Initialize and load a model with optional pretrained weights."""
        model = JPFormer(
            enc_in=config.enc_in,
            dec_in=config.dec_in,
            c_out=config.c_out,
            seq_len=config.seq_len,
            label_len=config.label_len,
            out_len=config.pred_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            d_layers=config.d_layers,
            d_ff=config.d_ff,
            factor=config.factor,
            dropout=config.dropout,
            embed=config.embed,
            activation=config.activation,
            output_attention=config.output_attention,
            mix=config.mix,
            device=device
        ).float().to(device)

        model_name = model.__class__.__name__
        print(f"Initialised model: {model_name}")

        # load pre-trained weights if provided
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                model.load_state_dict(torch.load(
                    pretrained_weights_path, map_location=device))
                print(
                    f"Successfully loaded pretrained weights from: {pretrained_weights_path}")
            except Exception as e:
                print(f"Error loading pretrained weights: {str(e)}")
                print("Continuing with random initialisation...")
        else:
            print(
                f"Pretrained weights file not found or not provided. Using random initialisation.")

        return model, model_name

    def inference_loop(self, encoder_input, decoder_input, last_timestamp, model, device):
        """Perform inference"""
        #  inference
        with torch.no_grad():
            encoder_input = encoder_input.unsqueeze(0).to(device)
            decoder_input = decoder_input.unsqueeze(0).to(device)
            output = model(encoder_input, decoder_input)

        # process the output
        output_df = pd.DataFrame(
            output.cpu().numpy().squeeze(), columns=['glucose_value'])
        # add timestamp to output_df based on last_timestamp + 5 minutes and 24
        # increments of 5 minutes
        output_df['timestamp'] = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=5), periods=24, freq='5min')

        # denormalised output
        output_df['glucose_value'] = (
            output_df['glucose_value'] * 70.27050122812615) + 152.91051040286524

        # find first index below 70mg/dl
        hypo_index = output_df[output_df['glucose_value'] < 70].index

        if hypo_index.empty:
            hypo_time = None
        else:
            hypo_time = output_df['timestamp'].iloc[hypo_index[0]]

        hyper_index = output_df[output_df['glucose_value'] > 180].index
        if hyper_index.empty:
            hyper_time = None
        else:
            hyper_time = output_df['timestamp'].iloc[hyper_index[0]]

        return hypo_time, hyper_time, output_df

    def get_glucose_status(self, value):
        """Determine the status and color based on glucose value"""
        if value < 70:
            return "HYPOGLYCAEMIA", self.HYPO_HEX
        elif value > 180:
            return "HYPERGLYCAEMIA", self.HYPER_HEX
        else:
            return "NORMAL", self.NORMAL_HEX

    def on_closing(self):
        """Handle application closing"""
        self.running = False
        time.sleep(0.5)  # Give threads time to close
        self.root.destroy()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Glucose Monitoring Dashboard')
    parser.add_argument('--ptid', type=int, default=540,
                        help='patient id number')
    parser.add_argument('--optimised_for', type=str, default='hypo',
                        help="optimised for 'hypo' EP% or 'overall' EP%")
    args = parser.parse_args()

    # create Tkinter window
    root = tk.Tk()
    root.title("Blood Glucose Monitoring Dashboard")

    # set window to open maximised (works on windows not mac)
    root.state('zoomed')

    app = GlucoseDashboard(root, ptid=args.ptid,
                           optimised_for=args.optimised_for)

    # set up the close handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
