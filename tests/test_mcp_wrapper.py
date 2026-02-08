import os
import sys
import unittest
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools.mcp_wrapper import MCPToolFactory, MCPToolRegistry


class TestMCPToolFactory(unittest.TestCase):

    def test_create_tool_class(self):
        """Test that factory creates tool class with correct attributes"""
        config = {
            'name': 'test_tool',
            'description': 'Test description',
            'endpoint_url': 'https://example.com/sse',
            'parameters': [
                {
                    'name': 'query',
                    'type': 'string',
                    'description': 'Test parameter',
                    'required': True
                }
            ]
        }
        
        tool_class = MCPToolFactory.create_tool_class('test_tool', config)
        
        self.assertEqual(tool_class.__name__, 'test_tool')
        self.assertEqual(tool_class.description, 'Test description')
        self.assertEqual(tool_class.parameters, config['parameters'])

    def test_tool_initialization(self):
        """Test that tool instance has correct endpoint_url"""
        config = {
            'name': 'test_tool',
            'description': 'Test description',
            'endpoint_url': 'https://example.com/sse',
            'parameters': []
        }
        
        tool_class = MCPToolFactory.create_tool_class('test_tool', config)
        tool_instance = tool_class()
        
        self.assertEqual(tool_instance.endpoint_url, 'https://example.com/sse')

    def test_env_var_substitution(self):
        """Test environment variable substitution in endpoint_url"""
        os.environ['TEST_MCP_BASE'] = 'https://test.example.com'
        
        config = {
            'name': 'test_tool',
            'description': 'Test description',
            'endpoint_url': '${TEST_MCP_BASE}/mcp/sse',
            'parameters': []
        }
        
        tool_class = MCPToolFactory.create_tool_class('test_tool', config)
        tool_instance = tool_class()
        
        self.assertEqual(tool_instance.endpoint_url, 'https://test.example.com/mcp/sse')
        
        del os.environ['TEST_MCP_BASE']

    def test_missing_env_var_raises_error(self):
        """Test that missing environment variable raises ValueError"""
        config = {
            'name': 'test_tool',
            'description': 'Test description',
            'endpoint_url': '${UNDEFINED_VAR}/mcp/sse',
            'parameters': []
        }
        
        tool_class = MCPToolFactory.create_tool_class('test_tool', config)
        
        with self.assertRaises(ValueError) as context:
            tool_instance = tool_class()
        
        self.assertIn('UNDEFINED_VAR', str(context.exception))

    def test_custom_config_override(self):
        """Test that custom cfg parameter overrides endpoint_url"""
        config = {
            'name': 'test_tool',
            'description': 'Test description',
            'endpoint_url': 'https://default.com/sse',
            'parameters': []
        }
        
        tool_class = MCPToolFactory.create_tool_class('test_tool', config)
        tool_instance = tool_class(cfg={'endpoint_url': 'https://custom.com/sse'})
        
        self.assertEqual(tool_instance.endpoint_url, 'https://custom.com/sse')


class TestMCPToolRegistry(unittest.TestCase):

    def test_load_config(self):
        """Test loading valid YAML configuration"""
        config_content = """
mcp_tools:
  test_tool:
    name: 'test_tool'
    description: 'Test tool'
    endpoint_url: 'https://example.com/sse'
    parameters:
      - name: 'query'
        type: 'string'
        description: 'Test parameter'
        required: true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            registry = MCPToolRegistry(temp_path)
            config = registry.load_config()
            
            self.assertIn('mcp_tools', config)
            self.assertIn('test_tool', config['mcp_tools'])
            self.assertEqual(config['mcp_tools']['test_tool']['name'], 'test_tool')
        finally:
            os.unlink(temp_path)

    def test_missing_config_file(self):
        """Test that missing config file raises FileNotFoundError"""
        registry = MCPToolRegistry('/nonexistent/path/config.yaml')
        
        with self.assertRaises(FileNotFoundError):
            registry.load_config()

    def test_invalid_yaml(self):
        """Test that invalid YAML raises ValueError"""
        invalid_yaml = """
mcp_tools:
  test_tool:
    name: 'test_tool'
    description: [invalid yaml
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            registry = MCPToolRegistry(temp_path)
            
            with self.assertRaises(ValueError):
                registry.load_config()
        finally:
            os.unlink(temp_path)

    def test_validate_missing_required_fields(self):
        """Test that missing required fields raises ValueError"""
        invalid_config = """
mcp_tools:
  test_tool:
    name: 'test_tool'
    description: 'Test tool'
    endpoint_url: 'https://example.com/sse'
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config)
            temp_path = f.name
        
        try:
            registry = MCPToolRegistry(temp_path)
            
            with self.assertRaises(ValueError) as context:
                registry.load_config()
            
            self.assertIn('parameters', str(context.exception))
        finally:
            os.unlink(temp_path)

    def test_register_tools(self):
        """Test that tools are registered correctly"""
        config_content = """
mcp_tools:
  test_tool_1:
    name: 'test_tool_1'
    description: 'Test tool 1'
    endpoint_url: 'https://example.com/1/sse'
    parameters:
      - name: 'query'
        type: 'string'
        description: 'Test parameter'
        required: true
  test_tool_2:
    name: 'test_tool_2'
    description: 'Test tool 2'
    endpoint_url: 'https://example.com/2/sse'
    parameters: []
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            registry = MCPToolRegistry(temp_path)
            tools_dict = registry.register_tools()
            
            self.assertIn('test_tool_1', tools_dict)
            self.assertIn('test_tool_2', tools_dict)
            self.assertEqual(tools_dict['test_tool_1'].__name__, 'test_tool_1')
            self.assertEqual(tools_dict['test_tool_2'].__name__, 'test_tool_2')
        finally:
            os.unlink(temp_path)


class TestMCPIntegration(unittest.TestCase):

    def test_backward_compatibility(self):
        """Test that existing imports still work"""
        from app.tools import AmapMapsTool, EdgeOneTool
        
        self.assertIsNotNone(AmapMapsTool)
        self.assertIsNotNone(EdgeOneTool)
        self.assertEqual(AmapMapsTool.description, '高德地图位置搜索，查找地点、地址、POI等信息')
        self.assertEqual(EdgeOneTool.description, 'EdgeOne页面内容获取和位置相关信息搜索')

    def test_get_all_mcp_tools(self):
        """Test that get_all_mcp_tools returns dictionary"""
        from app.tools import get_all_mcp_tools
        
        tools = get_all_mcp_tools()
        
        self.assertIsInstance(tools, dict)
        self.assertIn('amap_maps', tools)
        self.assertIn('edgeone_search', tools)

    def test_tools_have_correct_endpoint_urls(self):
        """Test that tools have correct endpoint URLs from config"""
        from app.tools import AmapMapsTool, EdgeOneTool
        
        amap_tool = AmapMapsTool()
        edgeone_tool = EdgeOneTool()
        
        self.assertEqual(amap_tool.endpoint_url, 'https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse')
        self.assertEqual(edgeone_tool.endpoint_url, 'https://dashscope.aliyuncs.com/api/v1/mcps/EdgeOne/sse')


if __name__ == '__main__':
    unittest.main()
