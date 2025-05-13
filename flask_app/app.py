# Make sure we have all necessary imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import nltk
import traceback
import scipy.sparse as sp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Define the preprocessing function
def preprocess_comment(comment):
    """
    Apply preprocessing transformations to a comment.
    Handles various edge cases and ensures result is always a string.
    """
    try:
        # Handle None or non-string inputs
        if comment is None:
            return ""
            
        if not isinstance(comment, str):
            comment = str(comment)
            
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        # Ensure comment is not empty
        if not comment:
            return " "  # Return a space to prevent empty string issues
            
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        traceback.print_exc()
        return "" if comment is None else str(comment)

# Global variables for model and vectorizer
model = None
vectorizer = None

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer():
    """
    Load the ML model from MLflow and the vectorizer from local storage.
    Sets global variables for reuse across requests.
    
    Returns:
        bool: True if loading succeeded, False otherwise
    """
    global model, vectorizer
    try:
        # Set MLflow tracking URI to your server
        mlflow.set_tracking_uri("http://ec2-54-87-251-47.compute-1.amazonaws.com:5000/")
        print("Setting MLflow tracking URI")
        
        # Get the MlflowClient
        client = MlflowClient()
        print("MlflowClient created")
        
        # Define the model URI
        model_uri = "models:/yt_chrome_plugin_model/2"
        print(f"Loading model from {model_uri}")
        
        # Load the model from MLflow
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully")
        
        # Load the vectorizer from local file
        vectorizer_path = "./tfidf_vectorizer.pkl"
        print(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer loaded successfully")
        
        # Test that the vectorizer has the expected methods
        if not hasattr(vectorizer, 'transform') or not hasattr(vectorizer, 'get_feature_names_out'):
            raise AttributeError("Vectorizer is missing required methods")
            
        print("Model and vectorizer loaded and validated successfully")
        return True
    except FileNotFoundError as e:
        print(f"Error: Vectorizer file not found: {e}")
        traceback.print_exc()
        return False
    except ConnectionError as e:
        print(f"Error connecting to MLflow server: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Unexpected error loading model and vectorizer: {e}")
        traceback.print_exc()
        return False

# Helper function to convert sparse matrix to pandas DataFrame with expected schema
def sparse_matrix_to_dataframe(sparse_matrix, vectorizer):
    """
    Convert a sparse matrix from TF-IDF vectorizer to a pandas DataFrame with named columns.
    This ensures the input matches the schema expected by the MLflow model.
    
    Args:
        sparse_matrix: Sparse matrix from vectorizer.transform()
        vectorizer: The fitted TF-IDF vectorizer
        
    Returns:
        pandas DataFrame with named columns matching the expected schema
    """
    try:
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # For small matrices, we can convert to dense format
        if sparse_matrix.shape[0] < 100:  # Only for small number of samples
            # Convert sparse matrix to dense numpy array
            dense_array = sparse_matrix.toarray()
            
            # Create DataFrame with feature names as columns
            df = pd.DataFrame(dense_array, columns=feature_names)
            
        else:
            # For larger matrices, we'll create an empty DataFrame and fill only non-zero values
            df = pd.DataFrame(0, index=range(sparse_matrix.shape[0]), columns=feature_names)
            
            # Only process non-zero elements to save memory
            for i in range(sparse_matrix.shape[0]):
                # Get row as 1xN sparse matrix
                row = sparse_matrix[i]
                
                # Get indices and values of non-zero elements
                indices = row.indices
                values = row.data
                
                # Add values to DataFrame
                for idx, val in zip(indices, values):
                    if idx < len(feature_names):
                        col_name = feature_names[idx]
                        df.loc[i, col_name] = val
        
        return df
        
    except Exception as e:
        print(f"Error in sparse_matrix_to_dataframe: {e}")
        traceback.print_exc()
        # If conversion fails, try a simpler approach
        return pd.DataFrame(sparse_matrix.toarray(), columns=feature_names)

# Initialize the model and vectorizer at startup
load_model_and_vectorizer()

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """
    Endpoint to predict sentiment from comments with timestamps.
    Expects a JSON with a 'comments' field containing an array of objects,
    each with 'text' and 'timestamp' fields.
    Returns an array of predictions with the original comments and timestamps.
    """
    try:
        data = request.json
        comments_data = data.get('comments')
        
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        app.logger.info(f"Received prediction request with {len(comments_data)} comments and timestamps")

        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            app.logger.warning("Model or vectorizer not loaded, attempting to load")
            if not load_model_and_vectorizer():
                return jsonify({"error": "Failed to load model or vectorizer"}), 500

        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        app.logger.debug("Comments preprocessed successfully")
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        app.logger.debug(f"Vectorized comments shape: {transformed_comments.shape}")
        
        try:
            # Convert sparse matrix to DataFrame with expected schema
            input_df = sparse_matrix_to_dataframe(transformed_comments, vectorizer)
            app.logger.debug(f"Converted to DataFrame with shape: {input_df.shape}")
            
            # Make predictions using the DataFrame
            predictions = model.predict(input_df)
            app.logger.debug("Predictions made successfully")
            
        except Exception as e:
            app.logger.error(f"Error in prediction process: {e}")
            # Fallback: Try direct prediction if dataframe conversion fails
            try:
                app.logger.warning("Attempting direct prediction without DataFrame conversion")
                predictions = model.predict(transformed_comments)
                app.logger.info("Direct prediction successful")
            except Exception as direct_error:
                app.logger.error(f"Direct prediction also failed: {direct_error}")
                raise Exception(f"Both DataFrame and direct prediction methods failed: {e} | {direct_error}")
        
        # Convert predictions to strings for consistency
        predictions = [str(int(float(pred))) for pred in predictions]
        
        # Return the response with original comments, predicted sentiments, and timestamps
        response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
                  for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Prediction with timestamps failed: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict sentiment from comments.
    Expects a JSON with a 'comments' field containing an array of strings.
    Returns an array of predictions with the original comments.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        comments = data.get('comments')
        
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        app.logger.info(f"Received prediction request with {len(comments)} comments")

        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            app.logger.warning("Model or vectorizer not loaded, attempting to load")
            if not load_model_and_vectorizer():
                return jsonify({"error": "Failed to load model or vectorizer"}), 500

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        app.logger.debug("Comments preprocessed successfully")
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        app.logger.debug(f"Vectorized comments shape: {transformed_comments.shape}")
        
        try:
            # Convert sparse matrix to DataFrame with expected schema
            input_df = sparse_matrix_to_dataframe(transformed_comments, vectorizer)
            app.logger.debug(f"Converted to DataFrame with shape: {input_df.shape}")
            
            # Make predictions using the DataFrame
            predictions = model.predict(input_df)
            app.logger.debug("Predictions made successfully")
            
        except Exception as e:
            app.logger.error(f"Error in prediction process: {e}")
            # Fallback: Try direct prediction if dataframe conversion fails
            try:
                app.logger.warning("Attempting direct prediction without DataFrame conversion")
                predictions = model.predict(transformed_comments)
                app.logger.info("Direct prediction successful")
            except Exception as direct_error:
                app.logger.error(f"Direct prediction also failed: {direct_error}")
                raise Exception(f"Both DataFrame and direct prediction methods failed: {e} | {direct_error}")
        
        # Convert predictions to strings for consistency
        predictions = [str(int(float(pred))) for pred in predictions]
        
        # Return the response with original comments and predicted sentiments
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/debug_model', methods=['POST'])
def debug_model():
    """
    Debug endpoint that shows the model's input and output types.
    Useful for diagnosing schema mismatches.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        comments = data.get('comments')
        
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
            
        # Limited to 3 comments for safety
        comments = comments[:3]

        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            if not load_model_and_vectorizer():
                return jsonify({"error": "Failed to load model or vectorizer"}), 500

        # Preprocess each comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert to pandas DataFrame
        input_df = sparse_matrix_to_dataframe(transformed_comments, vectorizer)
        
        # Sample the DataFrame (first 5 columns)
        df_sample = input_df.iloc[:, :5].to_dict()
        
        # Get information about the model's input schema
        model_input_schema = str(model.metadata.get_input_schema()) if hasattr(model, 'metadata') else "Schema not available"
        
        # Return debug information
        debug_info = {
            "original_comments": comments,
            "preprocessed_comments": preprocessed_comments,
            "vectorized_shape": {
                "rows": transformed_comments.shape[0],
                "columns": transformed_comments.shape[1],
                "non_zero_elements": transformed_comments.nnz
            },
            "dataframe_sample": df_sample,
            "dataframe_shape": {
                "rows": input_df.shape[0],
                "columns": input_df.shape[1]
            },
            "model_input_schema": model_input_schema,
            "vectorizer_feature_names_sample": list(vectorizer.get_feature_names_out()[:10])
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint to check API status and model loading.
    Returns information about the model and vectorizer state.
    """
    status_info = {
        "api_status": "online",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "preprocessor_ready": stopwords is not None and WordNetLemmatizer is not None
    }
    
    # If model and vectorizer are loaded, add more information
    if model is not None and vectorizer is not None:
        try:
            status_info["vectorizer_features_count"] = len(vectorizer.get_feature_names_out())
            status_info["model_type"] = str(type(model))
        except Exception as e:
            status_info["error_fetching_details"] = str(e)
    
    status_code = 200 if model is not None and vectorizer is not None else 503
    
    return jsonify(status_info), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)