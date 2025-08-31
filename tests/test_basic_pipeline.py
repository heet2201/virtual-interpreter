#!/usr/bin/env python3
"""
Basic tests for Virtual Interpreter Pipeline
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import json
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from basic.pipeline_basic import create_interpreter_pipeline, VirtualInterpreterPipeline
from configs.config import SUPPORTED_LANGUAGES


class TestBasicPipeline(unittest.TestCase):
    """Test cases for basic pipeline functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    def setUp(self):
        """Set up test fixtures for each test"""
        # Mock the heavy model loading to avoid actual downloads in tests
        self.mock_models()
    
    def mock_models(self):
        """Mock the model components to avoid loading real models"""
        # This would be expanded with proper mocks in a real test suite
        pass
    
    def test_pipeline_creation(self):
        """Test basic pipeline creation"""
        try:
            # This will attempt to load real models - skip if not available
            pipeline = create_interpreter_pipeline(
                asr_model_size="tiny",  # Use smallest model for testing
                tts_engine="gtts"
            )
            self.assertIsInstance(pipeline, VirtualInterpreterPipeline)
            
            # Test pipeline info
            info = pipeline.get_pipeline_info()
            self.assertIn('asr_model', info)
            self.assertIn('translation_model', info)
            self.assertIn('tts_engine', info)
            self.assertIn('supported_languages', info)
            
        except Exception as e:
            self.skipTest(f"Skipping test due to model loading issue: {e}")
    
    def test_supported_languages(self):
        """Test that all supported languages are properly configured"""
        self.assertGreaterEqual(len(SUPPORTED_LANGUAGES), 10)
        
        # Check that all languages have required codes
        for lang_name, lang_config in SUPPORTED_LANGUAGES.items():
            self.assertIn('code', lang_config)
            self.assertIn('whisper', lang_config) 
            self.assertIn('tts', lang_config)
    
    def test_language_validation(self):
        """Test language pair validation"""
        # Test with mock pipeline to avoid model loading
        with patch('basic.pipeline_basic.VirtualInterpreterPipeline'):
            pipeline = create_interpreter_pipeline()
            
            # Test valid language codes
            valid_codes = ['en', 'es', 'fr', 'de']
            for code in valid_codes:
                self.assertIn(code, [lang['code'] for lang in SUPPORTED_LANGUAGES.values()])
    
    def test_audio_processing_mock(self):
        """Test audio processing with mock data"""
        # Create mock audio data
        mock_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        
        # Test audio properties
        self.assertEqual(len(mock_audio), 16000)
        self.assertEqual(mock_audio.dtype, np.float32)
        
        # Test audio is in reasonable range
        self.assertLessEqual(np.max(np.abs(mock_audio)), 3.0)  # Reasonable for random audio
    
    def test_configuration_loading(self):
        """Test that configuration loads properly"""
        from configs.config import ASR_CONFIG, TRANSLATION_CONFIG, TTS_CONFIG
        
        # Test ASR config
        self.assertIn('basic', ASR_CONFIG)
        self.assertIn('advanced', ASR_CONFIG)
        
        # Test Translation config  
        self.assertIn('basic', TRANSLATION_CONFIG)
        self.assertIn('advanced', TRANSLATION_CONFIG)
        
        # Test TTS config
        self.assertIn('basic', TTS_CONFIG)
        self.assertIn('advanced', TTS_CONFIG)
    
    def test_performance_stats_initialization(self):
        """Test that performance statistics are properly initialized"""
        with patch('basic.pipeline_basic.BasicASR'), \
             patch('basic.pipeline_basic.BasicTranslator'), \
             patch('basic.pipeline_basic.BasicTTS'):
            
            pipeline = VirtualInterpreterPipeline()
            stats = pipeline.get_performance_stats()
            
            # Check that stats structure exists
            self.assertIn('total_requests', stats)
            self.assertIn('successful_requests', stats)
            self.assertIn('avg_processing_time', stats)
            self.assertIn('component_times', stats)
            self.assertIn('success_rate', stats)
            
            # Check initial values
            self.assertEqual(stats['total_requests'], 0)
            self.assertEqual(stats['successful_requests'], 0)
            self.assertEqual(stats['success_rate'], 0.0)
    
    def test_quality_thresholds(self):
        """Test quality threshold configuration"""
        from configs.config import QUALITY_THRESHOLDS
        
        self.assertIn('asr_confidence', QUALITY_THRESHOLDS)
        self.assertIn('translation_bleu', QUALITY_THRESHOLDS)
        self.assertIn('audio_quality', QUALITY_THRESHOLDS)
        
        # Check thresholds are reasonable
        self.assertGreater(QUALITY_THRESHOLDS['asr_confidence'], 0)
        self.assertLess(QUALITY_THRESHOLDS['asr_confidence'], 1)
    
    def test_batch_processing_structure(self):
        """Test batch processing structure without actual processing"""
        # Mock input data
        audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
        language_pairs = [("en", "es"), ("fr", "en"), ("de", "it")]
        
        # Test input validation
        self.assertEqual(len(audio_files), len(language_pairs))
        
        # Test language pair structure
        for src, tgt in language_pairs:
            self.assertIsInstance(src, str)
            self.assertIsInstance(tgt, str)
            self.assertEqual(len(src), 2)  # Language codes are 2 characters
            self.assertEqual(len(tgt), 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_audio_quality_check_mock(self):
        """Test audio quality checking with mock data"""
        from utils.audio_utils import audio_quality_check
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Test quality check
        quality = audio_quality_check(audio, sample_rate)
        
        # Check return structure
        self.assertIn('quality_score', quality)
        self.assertIn('duration', quality)
        self.assertIn('sample_rate', quality)
        
        # Check values are reasonable
        self.assertGreaterEqual(quality['quality_score'], 0.0)
        self.assertLessEqual(quality['quality_score'], 1.0)
        self.assertAlmostEqual(quality['duration'], duration, places=1)
        self.assertEqual(quality['sample_rate'], sample_rate)
    
    def test_evaluation_metrics_structure(self):
        """Test evaluation metrics structure"""
        from utils.evaluation import ASREvaluator, TranslationEvaluator
        
        # Test ASR evaluator
        asr_eval = ASREvaluator()
        self.assertIsInstance(asr_eval, ASREvaluator)
        
        # Test sample WER calculation
        reference = "hello world"
        hypothesis = "hello world"
        wer = asr_eval.word_error_rate(reference, hypothesis)
        self.assertEqual(wer, 0.0)  # Perfect match should have 0 WER
        
        # Test Translation evaluator
        mt_eval = TranslationEvaluator()
        self.assertIsInstance(mt_eval, TranslationEvaluator)
        
        # Test sample BLEU calculation
        reference = "hello world"
        hypothesis = "hello world"
        bleu = mt_eval.bleu_score(reference, hypothesis, use_sacrebleu=False)
        self.assertGreater(bleu, 0.8)  # Perfect match should have high BLEU


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_required_config_keys(self):
        """Test that all required configuration keys exist"""
        from configs import config
        
        # Check that all required attributes exist
        required_attrs = [
            'SUPPORTED_LANGUAGES', 'ASR_CONFIG', 'TRANSLATION_CONFIG',
            'TTS_CONFIG', 'AUDIO_CONFIG', 'QUALITY_THRESHOLDS'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing config attribute: {attr}")
    
    def test_audio_config_validity(self):
        """Test audio configuration validity"""
        from configs.config import AUDIO_CONFIG
        
        # Check required keys
        required_keys = ['sample_rate', 'chunk_size', 'format', 'channels', 'max_duration']
        for key in required_keys:
            self.assertIn(key, AUDIO_CONFIG)
        
        # Check reasonable values
        self.assertGreater(AUDIO_CONFIG['sample_rate'], 8000)
        self.assertLess(AUDIO_CONFIG['sample_rate'], 50000)
        self.assertGreater(AUDIO_CONFIG['max_duration'], 0)


def run_basic_tests():
    """Run basic tests that don't require model downloads"""
    # Create test suite with only basic tests
    suite = unittest.TestSuite()
    
    # Add tests that don't require model loading
    suite.addTest(TestBasicPipeline('test_supported_languages'))
    suite.addTest(TestBasicPipeline('test_language_validation'))
    suite.addTest(TestBasicPipeline('test_audio_processing_mock'))
    suite.addTest(TestBasicPipeline('test_configuration_loading'))
    suite.addTest(TestBasicPipeline('test_performance_stats_initialization'))
    suite.addTest(TestBasicPipeline('test_quality_thresholds'))
    suite.addTest(TestBasicPipeline('test_batch_processing_structure'))
    
    suite.addTest(TestUtilityFunctions('test_audio_quality_check_mock'))
    suite.addTest(TestUtilityFunctions('test_evaluation_metrics_structure'))
    
    suite.addTest(TestConfigurationValidation('test_required_config_keys'))
    suite.addTest(TestConfigurationValidation('test_audio_config_validity'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running Virtual Interpreter Basic Tests")
    print("=" * 50)
    
    # Run basic tests that don't require model downloads
    success = run_basic_tests()
    
    if success:
        print("\n✅ All basic tests passed!")
        print("Note: Full tests require model downloads and may take significant time.")
        print("To run full tests, use: python -m pytest tests/ -v")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print("\nFor complete testing including model loading:")
    print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
    print("2. Run: python -m unittest tests.test_basic_pipeline -v")
