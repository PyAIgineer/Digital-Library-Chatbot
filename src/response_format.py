"""
Modular response formatter utilizing LLM capabilities and specialized libraries.
This module provides a robust way to transform educational content into various formats.
"""

import re
import pandas as pd
import markdown
from typing import List, Dict, Any, Optional, Union, Callable
from functools import lru_cache

# Define formatter protocol/interface
class FormatterInterface:
    """Base interface for all formatters."""
    
    def format(self, content: str, **kwargs) -> str:
        """Format the content according to specific rules."""
        raise NotImplementedError("Subclasses must implement format method")
    
    def can_handle(self, content: str) -> bool:
        """Check if this formatter can handle the given content."""
        return True

class LLMFormatterMixin:
    """Mixin to provide LLM-based formatting capabilities."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        self.llm = llm
    
    def format_with_llm(self, content: str, prompt_template: str, **kwargs) -> str:
        """Use LLM to format content based on a prompt template."""
        if not self.llm:
            return content
            
        try:
            # Fill prompt template with content and any additional kwargs
            prompt_kwargs = {"content": content, **kwargs}
            prompt = prompt_template.format(**prompt_kwargs)
            
            # Generate formatted content with LLM
            response = self.llm.invoke(prompt)
            
            # Handle different response types (string or message object)
            if hasattr(response, 'content'):
                # Message-style response (like from ChatGroq)
                return response.content
            else:
                # String response
                return str(response)
        except Exception as e:
            print(f"LLM formatting error: {str(e)}")
            return content

# Individual formatters

class TableFormatter(FormatterInterface, LLMFormatterMixin):
    """Format content as a table."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        LLMFormatterMixin.__init__(self, llm)
        self.table_prompt = """
        Convert the following text into a well-structured table.
        If the content contains structured data like key-value pairs, create columns for those.
        If not, organize the content in a logical way that makes it easy to understand.
        
        CONTENT:
        {content}
        
        {additional_instructions}
        
        Return only a markdown table, nothing else.
        """
    
    def can_handle(self, content: str) -> bool:
        """Check if content is suitable for table formatting."""
        # Look for structure like key-value pairs or lists
        if re.search(r':[^:]+(?:\n|$)', content):
            return True
        if re.search(r'^\s*[-•*]\s+', content, re.MULTILINE):
            return True
        # Content too short probably won't make a good table
        return len(content) > 100
    
    def format(self, content: str, **kwargs) -> str:
        """Format content as a table."""
        # Try to use LLM if available
        if self.llm:
            additional_instructions = kwargs.get("additional_instructions", "")
            result = self.format_with_llm(content, self.table_prompt, additional_instructions=additional_instructions)
            if result != content and "Markdown tables use | to separate columns" not in result:
                return result
        
        # Fallback to pandas-based table creation
        return self._create_table_with_pandas(content)
    
    def _create_table_with_pandas(self, content: str) -> str:
        """Use pandas to create a table from structured content."""
        lines = content.split('\n')
        data = []
        
        # Try to identify key-value pairs
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Look for key-value patterns
            kv_match = re.match(r'^([^:]+):\s*(.+)$', line)
            if kv_match:
                data.append([kv_match.group(1).strip(), kv_match.group(2).strip()])
                continue
                
            # Look for list items
            list_match = re.match(r'^\s*[-•*]\s+(.+)$', line)
            if list_match:
                item = list_match.group(1).strip()
                # Try to split on obvious separators
                parts = re.split(r'\s*[-:]\s+|\s{3,}', item, 1)
                if len(parts) > 1:
                    data.append([parts[0].strip(), parts[1].strip()])
                else:
                    data.append([item, ""])
        
        # If not enough data for a table, create some structure
        if len(data) < 2:
            # Try to break content into sections and use them as rows
            paragraphs = content.split('\n\n')
            if len(paragraphs) >= 2:
                data = []
                for i, para in enumerate(paragraphs[:5], 1):  # Limit to 5 rows
                    if para.strip():
                        # Use first sentence as key, rest as value
                        first_sentence_match = re.match(r'^([^.!?]+[.!?])\s*(.*)$', para.strip(), re.DOTALL)
                        if first_sentence_match:
                            key = first_sentence_match.group(1).strip()
                            value = first_sentence_match.group(2).strip()
                            data.append([key, value])
                        else:
                            data.append([f"Section {i}", para.strip()])
        
        # If we have data, create a table
        if data:
            # Create DataFrame
            df = pd.DataFrame(data)
            headers = ["Topic", "Details"] if len(df.columns) == 2 else [f"Column {i+1}" for i in range(len(df.columns))]
            df.columns = headers
            
            # Convert to markdown
            markdown_table = df.to_markdown(index=False)
            return markdown_table
        
        # Fallback if all else fails
        return f"```\n{content}\n```"


class BulletPointFormatter(FormatterInterface, LLMFormatterMixin):
    """Format content as bullet points."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        LLMFormatterMixin.__init__(self, llm)
        self.bullet_prompt = """
        Convert the following text into a well-structured bullet point list.
        Use hierarchical structure with main points and supporting details.
        Group related information together.
        
        CONTENT:
        {content}
        
        Return only the bullet point list, nothing else.
        """
    
    def format(self, content: str, **kwargs) -> str:
        """Format content as bullet points."""
        # Use LLM if available
        if self.llm:
            result = self.format_with_llm(content, self.bullet_prompt)
            if result != content:
                return result
        
        # Fallback to manual bullet point creation
        return self._create_bullet_points(content)
    
    def _create_bullet_points(self, content: str) -> str:
        """Create bullet points from content."""
        paragraphs = re.split(r'\n\s*\n', content)
        result = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Check if already formatted as bullet
            if re.match(r'^\s*[•\-\*]', paragraph):
                result.append(paragraph)
                continue
                
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            # Group short sentences 
            current_group = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                current_group.append(sentence)
                
                # Create a bullet after 1-2 sentences or at the end
                if len(current_group) >= 2 or sentence == sentences[-1]:
                    bullet_text = " ".join(current_group).strip()
                    if bullet_text:
                        result.append(f"* {bullet_text}")
                    current_group = []
        
        return "\n".join(result)


class PresentationFormatter(FormatterInterface, LLMFormatterMixin):
    """Format content as presentation slides."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        LLMFormatterMixin.__init__(self, llm)
        self.presentation_prompt = """
        Convert the following text into a presentation format with slides.
        Create a title slide and organize the content into logical sections.
        Use clear headings and bullet points for each slide.
        Aim for approximately {slide_count} slides total.
        
        CONTENT:
        {content}
        
        Use markdown formatting with # for the presentation title and ## for slide headings.
        """
    
    def format(self, content: str, **kwargs) -> str:
        """Format content as presentation slides."""
        slide_count = kwargs.get("slide_count", 5)
        
        # Use LLM if available
        if self.llm:
            result = self.format_with_llm(content, self.presentation_prompt, slide_count=slide_count)
            if result != content:
                return result
        
        # Fallback to manual presentation formatting
        return self._create_presentation(content, slide_count)
    
    def _create_presentation(self, content: str, slide_count: int = 5) -> str:
        """Create presentation slides from content."""
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Extract title
        title = "Presentation"
        if paragraphs and len(paragraphs[0].strip()) < 80:
            title = paragraphs[0].strip()
            paragraphs = paragraphs[1:]
        
        result = [f"# {title}\n"]
        
        # Calculate paragraphs per slide to meet target slide count
        target_slides = min(slide_count, max(1, len(paragraphs) // 2))
        paras_per_slide = max(1, len(paragraphs) // target_slides)
        
        # Create slides
        current_slide = []
        slide_title = None
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            # Check if this could be a slide title
            if len(para.strip()) < 60 and not para.strip().endswith(('.', '!', '?')):
                # This looks like a title - start new slide
                if current_slide:
                    slide_content = "\n".join(current_slide)
                    if slide_title:
                        result.append(f"## {slide_title}\n\n{slide_content}")
                    else:
                        result.append(f"## Slide {len(result)}\n\n{slide_content}")
                    current_slide = []
                
                slide_title = para.strip()
            else:
                # Regular content
                current_slide.append(para)
                
                # Check if we should start a new slide
                if len(current_slide) >= paras_per_slide:
                    slide_content = "\n".join(current_slide)
                    if slide_title:
                        result.append(f"## {slide_title}\n\n{slide_content}")
                    else:
                        result.append(f"## Slide {len(result)}\n\n{slide_content}")
                    current_slide = []
                    slide_title = None
        
        # Add final slide if needed
        if current_slide:
            slide_content = "\n".join(current_slide)
            if slide_title:
                result.append(f"## {slide_title}\n\n{slide_content}")
            else:
                result.append(f"## Slide {len(result)}\n\n{slide_content}")
        
        return "\n\n".join(result)


class ComparisonFormatter(FormatterInterface, LLMFormatterMixin):
    """Format content as a comparison."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        LLMFormatterMixin.__init__(self, llm)
        self.comparison_prompt = """
        Reformat the following text as a clear comparative analysis.
        Identify the main entities being compared and structure the comparison with clear headings.
        Highlight similarities and differences in a structured way.
        
        CONTENT:
        {content}
        
        {aspect_instructions}
        
        Use markdown formatting with headings and bullet points for clarity.
        """
    
    def format(self, content: str, **kwargs) -> str:
        """Format content as a comparison."""
        # Get comparison aspects if provided
        aspects = kwargs.get("comparison_aspects", [])
        aspect_instructions = ""
        if aspects:
            aspect_instructions = f"Focus on comparing these specific aspects: {', '.join(aspects)}"
        
        # Use LLM if available
        if self.llm:
            result = self.format_with_llm(content, self.comparison_prompt, aspect_instructions=aspect_instructions)
            if result != content:
                return result
        
        # Fallback to manual comparison formatting
        return self._create_comparison(content, aspects)
    
    def _create_comparison(self, content: str, aspects: List[str] = None) -> str:
        """Create a comparison from content."""
        # Look for comparison indicators
        comparison_terms = ["versus", "vs.", "compared to", "in contrast to", "unlike", 
                          "similar to", "while", "whereas", "on the other hand"]
        
        # Check if this is a comparison
        contains_comparison = any(term in content.lower() for term in comparison_terms)
        
        if not contains_comparison:
            # Not a comparison, just add a header
            return f"## Comparative Analysis\n\n{content}"
        
        # Extract entities being compared
        entities = set()
        for term in comparison_terms:
            pattern = rf'(.+?)\s+{term}\s+(.+?)[\.\,]'
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    entities.update([m.strip() for m in match])
        
        # Start building the comparison
        result = ["# Comparative Analysis\n"]
        
        # Add entities if found
        if len(entities) >= 2:
            result.append("## Entities Compared\n")
            result.append(", ".join(list(entities)[:3]))
        
        # Add aspects if provided
        if aspects:
            result.append("\n## Comparison Aspects\n")
            for aspect in aspects:
                result.append(f"* {aspect}")
        
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Process paragraphs looking for comparison content
        current_section = []
        current_aspect = None
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Check if paragraph contains comparison terms
            para_has_comparison = any(term in para.lower() for term in comparison_terms)
            
            # Check if paragraph starts with what looks like an aspect heading
            aspect_match = re.match(r'^([A-Z][^.!?:]{3,50})[:.]\s*(.+)$', para, re.DOTALL)
            
            if aspect_match and (aspects is None or any(a.lower() in aspect_match.group(1).lower() for a in aspects)):
                # This looks like a new aspect - add previous section if exists
                if current_section:
                    if current_aspect:
                        result.append(f"\n## {current_aspect}\n\n" + "\n".join(current_section))
                    else:
                        result.append("\n## Comparison Point\n\n" + "\n".join(current_section))
                    current_section = []
                
                # Set new aspect
                current_aspect = aspect_match.group(1)
                # Add the rest of the paragraph to the section
                current_section.append(aspect_match.group(2))
            elif para_has_comparison:
                # Contains comparison terms - add to current section
                current_section.append(para)
            else:
                # Regular paragraph - check if it belongs to current section
                current_section.append(para)
                
                # If section is getting long, finish it
                if len("\n".join(current_section)) > 500:
                    if current_aspect:
                        result.append(f"\n## {current_aspect}\n\n" + "\n".join(current_section))
                    else:
                        result.append("\n## Comparison Point\n\n" + "\n".join(current_section))
                    current_section = []
                    current_aspect = None
        
        # Add final section if exists
        if current_section:
            if current_aspect:
                result.append(f"\n## {current_aspect}\n\n" + "\n".join(current_section))
            else:
                result.append("\n## Comparison Point\n\n" + "\n".join(current_section))
        
        return "\n".join(result)


class StepByStepFormatter(FormatterInterface, LLMFormatterMixin):
    """Format content as step-by-step guide."""
    
    def __init__(self, llm=None):
        """Initialize with an optional LLM."""
        LLMFormatterMixin.__init__(self, llm)
        self.steps_prompt = """
        Convert the following text into a clear step-by-step guide.
        Break down the process into sequential steps with clear numbering.
        {example_instruction}
        
        CONTENT:
        {content}
        
        Use markdown formatting with headings for each step.
        """
    
    def format(self, content: str, **kwargs) -> str:
        """Format content as step-by-step guide."""
        # Check if examples are requested
        include_examples = kwargs.get("include_examples", True)
        example_instruction = "Include brief examples for each step." if include_examples else ""
        
        # Use LLM if available
        if self.llm:
            result = self.format_with_llm(content, self.steps_prompt, example_instruction=example_instruction)
            if result != content:
                return result
        
        # Fallback to manual step formatting
        return self._create_steps(content)
    
    def _create_steps(self, content: str) -> str:
        """Create step-by-step guide from content."""
        # Check if content already has steps
        has_steps = False
        step_pattern = re.compile(r'^\s*(\d+\.|step\s+\d+:?|[•\-\*])', re.IGNORECASE)
        
        lines = content.split('\n')
        for line in lines:
            if step_pattern.match(line):
                has_steps = True
                break
        
        result = ["# Step-by-Step Guide\n"]
        
        if has_steps:
            # Content already has steps - reorganize them
            current_step = []
            step_num = 0
            
            for line in lines:
                if step_pattern.match(line):
                    # Save previous step if exists
                    if current_step:
                        result.append(f"## Step {step_num}\n\n" + "\n".join(current_step))
                    
                    # Start new step
                    step_num += 1
                    # Remove original step marker
                    cleaned_line = step_pattern.sub('', line).strip()
                    current_step = [cleaned_line]
                elif line.strip():
                    # Add to current step
                    current_step.append(line)
            
            # Add final step
            if current_step:
                result.append(f"## Step {step_num}\n\n" + "\n".join(current_step))
        else:
            # No explicit steps - create steps from paragraphs
            paragraphs = re.split(r'\n\s*\n', content)
            
            # Try to identify a process sequence
            sequence_candidates = []
            for i, para in enumerate(paragraphs):
                if re.search(r'\b(first|second|third|next|then|finally|lastly)\b', para, re.IGNORECASE):
                    sequence_candidates.append(i)
            
            # If we identified a sequence, organize paragraphs as steps
            if len(sequence_candidates) >= 2:
                # Organize around sequence indicators
                for i, para in enumerate(paragraphs):
                    if not para.strip():
                        continue
                    
                    # Number steps starting from sequence indicators
                    if i in sequence_candidates or i == 0:
                        step_num = sequence_candidates.index(i) + 1 if i in sequence_candidates else 1
                        result.append(f"## Step {step_num}\n\n{para}")
            else:
                # No sequence indicators - split content more evenly
                for i, para in enumerate(paragraphs, 1):
                    if not para.strip():
                        continue
                    
                    result.append(f"## Step {i}\n\n{para}")
        
        return "\n\n".join(result)


class ResponseFormatter:
    """Main formatter class that uses appropriate formatters based on content and request."""
    
    def __init__(self, llm=None):
        """Initialize with formatters and optional LLM."""
        self.llm = llm
        
        # Initialize formatters
        self.formatters = {
            "tabular": TableFormatter(llm),
            "bullets": BulletPointFormatter(llm),
            "presentation": PresentationFormatter(llm),
            "comparison": ComparisonFormatter(llm),
            "steps": StepByStepFormatter(llm)
        }
        
        # Format aliases
        self.format_aliases = {
            "table": "tabular",
            "bullet_points": "bullets",
            "bullet": "bullets",
            "slides": "presentation",
            "comparative": "comparison",
            "step_by_step": "steps",
            "step": "steps"
        }
    
    def format_response(self, content: str, format_type: str, **kwargs) -> str:
        """Format content using the appropriate formatter."""
        # Resolve format alias
        format_type = format_type.lower()
        if format_type in self.format_aliases:
            format_type = self.format_aliases[format_type]
        
        # Get formatter
        formatter = self.formatters.get(format_type)
        
        if formatter and formatter.can_handle(content):
            return formatter.format(content, **kwargs)
        
        # Fallback if no formatter found or can't handle
        return content

@lru_cache(maxsize=5)
def extract_format_request(query: str):
    """Extract formatting request from a user query.
    
    Returns:
        Tuple of (cleaned_query, format_type, format_options)
    """
    format_indicators = {
        "tabular": ["in tabular format", "as a table", "in table format", "create a table", "show as table"],
        "bullets": ["in bullet points", "as bullet points", "as a bulleted list", "in bullet format", "list format"],
        "presentation": ["as presentation", "presentation format", "as slides", "in slide format", "as presentation slides"],
        "comparison": ["as a comparison", "in comparison format", "comparative format", "compare and contrast"],
        "steps": ["step by step", "as steps", "in steps", "step-by-step breakdown", "step by step guide"]
    }
    
    # Remove format request from query and determine format type
    cleaned_query = query
    format_type = None
    format_options = {}
    
    for fmt, indicators in format_indicators.items():
        for indicator in indicators:
            if indicator.lower() in query.lower():
                format_type = fmt
                # Remove formatting request from query
                cleaned_query = re.sub(f"(?i){indicator}", "", query).strip()
                break
        if format_type:
            break
    
    # Extract any additional format options
    if format_type == "tabular":
        # Check for columns/headers
        headers_match = re.search(r"with columns[:\s]+([^\.]+)", query, re.IGNORECASE)
        if headers_match:
            headers_text = headers_match.group(1).strip()
            headers = [h.strip() for h in re.split(r',|\band\b', headers_text)]
            format_options["headers"] = headers
            # Remove headers request from query
            cleaned_query = re.sub(r"(?i)with columns[:\s]+([^\.]+)", "", cleaned_query).strip()
    
    elif format_type == "presentation":
        # Check for slide count
        slides_match = re.search(r"(\d+)\s+slides", query, re.IGNORECASE)
        if slides_match:
            format_options["slide_count"] = int(slides_match.group(1))
    
    elif format_type == "comparison":
        # Check for aspects to compare
        aspects_match = re.search(r"comparing[:\s]+([^\.]+)", query, re.IGNORECASE)
        if aspects_match:
            aspects_text = aspects_match.group(1).strip()
            aspects = [a.strip() for a in re.split(r',|\band\b', aspects_text)]
            format_options["comparison_aspects"] = aspects
    
    return cleaned_query, format_type, format_options


# Helper function to initialize the formatter
def get_formatter(llm=None):
    """Get a formatter instance with the given LLM."""
    return ResponseFormatter(llm)


def format_with_llm(content: str, format_type: str, llm, **kwargs):
    """Format content using LLM for the heavy lifting."""
    formatter = get_formatter(llm)
    return formatter.format_response(content, format_type, **kwargs)