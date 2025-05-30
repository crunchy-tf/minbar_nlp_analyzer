# Minbar NLP Analyzer Service

This microservice performs Natural Language Processing (NLP) analysis on text data. It includes capabilities for:
-   Zero-shot sentiment analysis using custom healthcare-related labels.
-   Topic modeling using a pre-trained BERTopic model.
-   Keyword extraction.

The service can process data in batches via a scheduler or analyze individual documents on-demand via an API endpoint.

## API Endpoints

---

### Analysis Endpoint

#### `POST /analyze`
-   **Description**: Performs NLP analysis (sentiment, topic modeling, keyword extraction) on a single provided document's data. The results, including sentiment scores, assigned topics, and extracted keywords, are returned and also stored in a target PostgreSQL database.
-   **Request Body**: `NLPAnalysisRequest`
    -   `raw_mongo_id` (string, required): The original MongoDB ID of the document.
    -   `source` (string, required): The source type of the document (e.g., "post").
    -   `original_timestamp` (datetime string, required): Timestamp of the original document creation.
    -   `retrieved_by_keyword` (string, required): The keyword that led to the ingestion of this document.
    -   `keyword_language` (string, required): The language of the `retrieved_by_keyword`.
    -   `keyword_concept_id` (string, optional): The concept ID associated with the keyword.
    -   `detected_language` (string, optional): The language detected in the `cleaned_text`.
    -   `cleaned_text` (string, required): The preprocessed text content to be analyzed.
    -   `tokens_processed` (list of strings, optional): Processed tokens (e.g., after stopword removal) from the preprocessor.
    -   `lemmas` (list of strings, optional): Lemmatized tokens from the preprocessor.
    -   `original_url` (string, optional): URL of the original document, if available.
-   **Sample Request Body**:
    ```json
    {
      "raw_mongo_id": "680a75cf622ece5a1dc7a4bc",
      "source": "post",
      "original_timestamp": "2025-04-16T19:20:19Z",
      "retrieved_by_keyword": "تكاليف الأدوية",
      "keyword_language": "ar",
      "keyword_concept_id": "670a12bb611fff9a1ab6b3ac",
      "detected_language": "ar",
      "cleaned_text": "في إطار زيارته الحالية لدولة تونس وزير الاستثمار يبحث مع الجانب التونسي سبل تعزيز التعاون المشترك في مختلف المجالات.",
      "tokens_processed": ["إطار", "زيارته", "الحالية", "لدولة", "تونس", "وزير", "الاستثمار", "يبحث", "الجانب", "التونسي", "سبل", "تعزيز", "التعاون", "المشترك", "مختلف", "المجالات"],
      "lemmas": ["إِطَار", "زِيَارَة", "حَالِيّ", "دَوْلَة", "تُونِس", "وَزِير", "اِسْتِثْمَار", "بَحَثَ", "جَانِب", "تُونِسِيّ", "سَبِيل", "تَعْزِيز", "تَعَاوُن", "مُشْتَرَك", "مُخْتَلِف", "مَجَال"],
      "original_url": "http://example.com/article/123"
    }
    ```
-   **Response Body**: `NLPAnalysisResponse`
    -   Contains the input fields along with:
        -   `processing_timestamp` (datetime): Timestamp of when this NLP analysis was performed.
        -   `overall_sentiment` (list of `SentimentScore`): List of sentiment labels and their scores.
        -   `assigned_topics` (list of `TopicInfo`): List of topics assigned to the document, including topic ID, name, keywords, and probability.
        -   `extracted_keywords_frequency` (list of `KeywordFrequency`): List of extracted keywords and their frequencies.
        -   `sentiment_on_extracted_keywords_summary` (list of `SentimentScore`, optional): Sentiment analysis performed on a concatenation of the top extracted keywords.
        -   `analysis_errors` (list of strings, optional): Any errors encountered during the analysis.

---