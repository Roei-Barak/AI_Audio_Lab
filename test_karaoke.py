"""
Comprehensive Test Suite for Karaoke Studio System
Tests all major components and workflows
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import logic to get pandas
import logic
import pandas as pd

import logic


class TestTaskManager(unittest.TestCase):
    """Test Task Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.task_manager = logic.TaskManager()
    
    def test_add_task(self):
        """Test adding a task to the manager"""
        task_id = self.task_manager.add_task("Test Song")
        self.assertIsNotNone(task_id)
        self.assertEqual(len(task_id), 8)
        self.assertIn(task_id, self.task_manager.tasks)
        self.assertEqual(self.task_manager.tasks[task_id].song_name, "Test Song")
        self.assertEqual(self.task_manager.tasks[task_id].status, "⏳")
        self.assertEqual(self.task_manager.tasks[task_id].progress, 0)
    
    def test_update_task_status(self):
        """Test updating task status"""
        task_id = self.task_manager.add_task("Test Song")
        self.task_manager.update_task(task_id, status="📥", progress=25)
        self.assertEqual(self.task_manager.tasks[task_id].status, "📥")
        self.assertEqual(self.task_manager.tasks[task_id].progress, 25)
    
    def test_update_task_progress(self):
        """Test updating only task progress"""
        task_id = self.task_manager.add_task("Test Song")
        self.task_manager.update_task(task_id, progress=50)
        self.assertEqual(self.task_manager.tasks[task_id].progress, 50)
        self.assertEqual(self.task_manager.tasks[task_id].status, "⏳")
    
    def test_get_dataframe(self):
        """Test getting tasks as dataframe"""
        self.task_manager.add_task("Song 1")
        self.task_manager.add_task("Song 2")
        df = self.task_manager.get_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("ID", df.columns)
        self.assertIn("שיר", df.columns)
        self.assertIn("סטטוס", df.columns)
    
    def test_remove_task(self):
        """Test removing a task"""
        task_id = self.task_manager.add_task("Test Song")
        self.assertIn(task_id, self.task_manager.tasks)
        self.task_manager.remove_task(task_id)
        self.assertNotIn(task_id, self.task_manager.tasks)
    
    def test_empty_dataframe(self):
        """Test dataframe when no tasks exist"""
        df = self.task_manager.get_dataframe()
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ["ID", "שיר", "סטטוס", "התקדמות", "VRAM", "זמן (s)"])


class TestResourceManager(unittest.TestCase):
    """Test Resource Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resource_manager = logic.ResourceManager(max_concurrent_heavy=2)
    
    def test_initialization(self):
        """Test resource manager initialization"""
        self.assertEqual(self.resource_manager.active_tasks, 0)
        self.assertIsNotNone(self.resource_manager.semaphore)
        self.assertIsNotNone(self.resource_manager.lock)
    
    def test_start_task(self):
        """Test starting a task"""
        self.assertEqual(self.resource_manager.active_tasks, 0)
        self.resource_manager.start_task()
        self.assertEqual(self.resource_manager.active_tasks, 1)
        self.resource_manager.start_task()
        self.assertEqual(self.resource_manager.active_tasks, 2)
    
    def test_end_task(self):
        """Test ending a task"""
        self.resource_manager.start_task()
        self.resource_manager.start_task()
        self.assertEqual(self.resource_manager.active_tasks, 2)
        self.resource_manager.end_task()
        self.assertEqual(self.resource_manager.active_tasks, 1)
    
    def test_get_status_idle(self):
        """Test status when idle"""
        status = self.resource_manager.get_status()
        self.assertIn("🟢", status)
        self.assertIn("פנוי", status)
    
    def test_get_status_active(self):
        """Test status when tasks are active"""
        self.resource_manager.start_task()
        status = self.resource_manager.get_status()
        self.assertIn("🟡", status)
        self.assertIn("משימה", status)
    
    def test_get_status_busy(self):
        """Test status when system is busy"""
        self.resource_manager.start_task()
        self.resource_manager.start_task()
        status = self.resource_manager.get_status()
        self.assertIn("🔴", status)
        self.assertIn("עמוס", status)


class TestBackendProcessor(unittest.TestCase):
    """Test Backend Processor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = logic.BackendProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Test special characters removal
        dirty_name = 'Song<>:"?*Name|.txt'
        clean_name = self.backend._sanitize_filename(dirty_name)
        self.assertEqual(clean_name, "SongName.txt")
        
        # Test empty string handling
        result = self.backend._sanitize_filename("   ")
        self.assertRegex(result, r"Video_\d+")
    
    def test_fmt_ass_time(self):
        """Test ASS time formatting"""
        # Test basic time formatting
        result = self.backend._fmt_ass_time(3665.5)  # 1 hour, 1 minute, 5.5 seconds
        self.assertEqual(result, "1:01:05.50")
        
        # Test zero
        result = self.backend._fmt_ass_time(0)
        self.assertEqual(result, "0:00:00.00")
        
        # Test invalid input
        result = self.backend._fmt_ass_time("invalid")
        self.assertEqual(result, "0:00:00.00")
    
    def test_log_function(self):
        """Test logging functionality"""
        logs = []
        msg = self.backend.log("Test message", logs)
        
        self.assertIn("Test message", msg)
        self.assertEqual(len(logs), 1)
        self.assertIn("Test message", logs[0])
    
    def test_log_without_list(self):
        """Test logging without log list"""
        msg = self.backend.log("Test message")
        self.assertIn("Test message", msg)
    
    def test_ass_to_dataframe_empty(self):
        """Test converting non-existent ASS file to dataframe"""
        df = self.backend.ass_to_dataframe("/nonexistent/file.ass")
        self.assertEqual(len(df), 0)
    
    def test_dataframe_to_ass(self):
        """Test converting dataframe to ASS file"""
        # Create test dataframe
        df = pd.DataFrame({
            "Start": ["0:00:00.00", "0:00:05.00"],
            "End": ["0:00:05.00", "0:00:10.00"],
            "Text": ["Hello", "World"]
        })
        
        output_path = os.path.join(self.temp_dir, "test.ass")
        self.backend.dataframe_to_ass(df, None, output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("Hello", content)
            self.assertIn("World", content)
            self.assertIn("[Script Info]", content)
    
    def test_update_ass_style(self):
        """Test ASS style updating"""
        # Create test ASS file
        ass_path = os.path.join(self.temp_dir, "style_test.ass")
        with open(ass_path, 'w', encoding='utf-8-sig') as f:
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("[V4+ Styles]\n")
            f.write("Style: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n")
        
        # Update style
        self.backend.update_ass_style(ass_path, 120, "#FF0000")
        
        # Verify changes
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("Arial,120", content)
            self.assertIn("&H000000FF", content)  # BGR format for red


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = logic.BackendProcessor()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ass_roundtrip(self):
        """Test converting ASS to dataframe and back"""
        # Create initial dataframe
        original_df = pd.DataFrame({
            "Start": ["0:00:00.00", "0:00:05.00", "0:00:10.00"],
            "End": ["0:00:05.00", "0:00:10.00", "0:00:15.00"],
            "Text": ["Line 1", "Line 2", "Line 3"]
        })
        
        # Convert to ASS
        ass_path = os.path.join(self.temp_dir, "roundtrip.ass")
        self.backend.dataframe_to_ass(original_df, None, ass_path)
        
        # Convert back to dataframe
        result_df = self.backend.ass_to_dataframe(ass_path)
        
        # Verify data integrity
        self.assertEqual(len(result_df), 3)
        self.assertListEqual(list(result_df["Text"]), ["Line 1", "Line 2", "Line 3"])
    
    def test_task_lifecycle(self):
        """Test complete task lifecycle"""
        task_manager = logic.TaskManager()
        
        # Add task
        task_id = task_manager.add_task("Test Song")
        self.assertIn(task_id, task_manager.tasks)
        
        # Simulate processing stages
        stages = ["📥", "🎚️", "📝", "🎬", "✅"]
        for i, stage in enumerate(stages):
            progress = (i + 1) * 20
            task_manager.update_task(task_id, status=stage, progress=progress)
            self.assertEqual(task_manager.tasks[task_id].status, stage)
            self.assertEqual(task_manager.tasks[task_id].progress, progress)
        
        # Get final dataframe
        df = task_manager.get_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["סטטוס"], "✅")
        self.assertIn("100%", df.iloc[0]["התקדמות"])
        
        # Remove task
        task_manager.remove_task(task_id)
        self.assertNotIn(task_id, task_manager.tasks)


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backend = logic.BackendProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_empty_dataframe_to_ass(self):
        """Test handling empty dataframe"""
        df = pd.DataFrame(columns=["Start", "End", "Text"])
        output_path = os.path.join(self.temp_dir, "empty.ass")
        
        self.backend.dataframe_to_ass(df, None, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # File should have header but no dialogue lines
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("[Script Info]", content)
    
    def test_special_characters_in_text(self):
        """Test handling special characters in ASS text"""
        df = pd.DataFrame({
            "Start": ["0:00:00.00"],
            "End": ["0:00:05.00"],
            "Text": ["עברית - English - 日本語"]
        })
        
        output_path = os.path.join(self.temp_dir, "special.ass")
        self.backend.dataframe_to_ass(df, None, output_path)
        
        result_df = self.backend.ass_to_dataframe(output_path)
        self.assertEqual(len(result_df), 1)


class TestResourceLimits(unittest.TestCase):
    """Test resource limitation and concurrency"""
    
    def test_semaphore_limit(self):
        """Test semaphore limits concurrent tasks"""
        rm = logic.ResourceManager(max_concurrent_heavy=2)
        
        # Acquire semaphore twice
        rm.semaphore.acquire()
        rm.semaphore.acquire()
        
        # Try to acquire third (should be blocking)
        acquired = rm.semaphore.acquire(blocking=False)
        self.assertFalse(acquired)
        
        # Release and try again
        rm.semaphore.release()
        acquired = rm.semaphore.acquire(blocking=False)
        self.assertTrue(acquired)


def run_all_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTaskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceManager))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceLimits))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
