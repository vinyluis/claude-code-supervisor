# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-22

### Added
- `SingleShotSupervisorAgent` class for single-execution workflows without feedback loops
- `FeedbackSupervisorAgent` class as explicit name for iterative feedback workflows  
- Tool selection capability for restricting Claude Code SDK tools

### Changed
- Improved workflow node naming for better clarity:
  - `collect_results` → `review_session` 
  - `validate_solution` → `test_and_analyze` (feedback) / `test_solution` (single-shot)
  - `provide_guidance` → `generate_guidance`
- SingleShotSupervisorAgent now properly reports success without iteration messages
- Updated all examples to explicitly use `FeedbackSupervisorAgent`
- Enhanced architecture with separated supervisor types and clearer responsibilities

### Fixed
- SingleShotSupervisorAgent incorrectly reporting "Maximum iterations reached" on success
- Test failures due to renamed workflow methods
- Context length management and message compression

## [0.1.3] - 2025-07-08

### Added
- AsyncIO error filtering to suppress harmless SDK cleanup messages
- Architecture refactoring with improved separation of concerns
- Enhanced testing infrastructure with better test coverage

### Changed
- Simplified supervisor structure and session control
- Improved error handling throughout the workflow
- Better integration between supervisor and Claude Code SDK

### Fixed
- Various utility function issues
- Session control and management improvements
- Documentation and example updates

## [0.1.2] - 2025-07-04

### Added
- Codebase integration flag for working with existing projects
- Credential passing through initialization kwargs
- Enhanced supervisor configuration management

### Changed
- Improved supervisor session control and lifecycle management
- Better error handling and recovery mechanisms
- Streamlined configuration system

### Fixed
- Example scripts and documentation
- Test suite reliability and coverage
- Configuration handling edge cases

## [0.1.1] - 2025-07-04

### Added
- Initial release of claude-code-supervisor
- Core `SupervisorAgent` class with intelligent feedback loops
- Integration with Claude Code SDK for automated problem-solving
- Support for multiple AI providers (Anthropic, AWS Bedrock, OpenAI)
- Comprehensive example suite demonstrating various use cases
- LLM-powered guidance generation for iterative improvement
- Session management and continuity across iterations
- Data processing workflows with input/output validation

### Features
- **Intelligent Feedback Loops**: Automatic analysis of Claude's output with targeted guidance
- **Multi-Provider Support**: Works with Anthropic API, AWS Bedrock, and OpenAI
- **Session Continuity**: Maintains context across multiple iterations
- **Test Integration**: Automated test execution and validation
- **Data Workflows**: Support for input/output data processing
- **Configurable Timeouts**: Flexible timeout and iteration limits
- **Error Recovery**: Intelligent error analysis and recovery suggestions
- **Context Management**: Automatic context length management and message reduction

### Examples
- Basic usage without input/output data
- List and dictionary processing workflows  
- CSV data processing simulation
- Custom prompts for OOP, performance, and data science patterns
- Integration with existing codebases

### Dependencies
- claude-code-sdk >= 0.0.14
- langchain >= 0.3.0
- langgraph >= 0.4.0  
- langchain-aws >= 0.2.0
- langchain-openai >= 0.3.0
- python-dotenv >= 0.19.0
- pytest >= 6.0.0