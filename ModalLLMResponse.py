import json
import os
import re
from openai import OpenAI
# from torch import res


class ModalLLMResponse:
    def __init__(
            self,
            api_key: str = None,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name: str = "qwen-vl-max"  # "qwen-vl-max" or "qwen3-vl-plus"
    ):
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set; create .env from .env.example")
        self.config = {
            "API_KEY": api_key,
            "API_BASE": base_url,
            "MODEL_NAME": model_name
        }
        self.client = OpenAI(api_key=self.config["API_KEY"], base_url=self.config["API_BASE"])

    def generate_response(self, image_base64: str, prompt: str) -> dict:
        completion = self.client.chat.completions.create(
            model=self.config["MODEL_NAME"],
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Parse response data
        # completion_data = json.loads(completion.model_dump_json())
        message_content = completion.choices[0].message.content
        # print(message_content)

        # Extract JSON content (handle nested braces correctly)
        start = message_content.find("{")
        end = message_content.rfind("}") + 1
        
        if start == -1 or end == 0:
            return {"error": "No JSON found in response", "raw_response": message_content[:200]}
        
        json_str = message_content[start:end].strip()
        # print(json_str)
        # Convert to dictionary and return
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response content", "raw_response": json_str[:200]}

class TextLLMResponse:
    """Text-only language model response class for scenarios that don't require image input"""
    def __init__(
            self,
            api_key: str = None,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name: str = "qwen3-max",  # Use text-only models such as qwen-plus, qwen-turbo, etc.
            timeout: float = 120.0  # 默认60秒超时
    ):
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set; create .env from .env.example")
        self.config = {
            "API_KEY": api_key,
            "API_BASE": base_url.strip().rstrip('/'),  # 移除末尾斜杠，避免路径问题
            "MODEL_NAME": model_name
        }
        self.timeout = timeout
        # 确保base_url格式正确，添加超时设置
        self.client = OpenAI(
            api_key=self.config["API_KEY"], 
            base_url=self.config["API_BASE"],
            timeout=timeout  # 添加超时设置
        )

    def generate_response(self, prompt: str, timeout: float = None) -> dict:
        """
        Accept only text prompt and return JSON-formatted response
        
        Args:
            prompt: Text prompt
            timeout: 可选的超时时间（秒），如果不指定则使用初始化时的默认值
            
        Returns:
            Parsed JSON dictionary, returns dictionary with error key if parsing fails
        """
        try:
            # 如果指定了超时时间，临时创建新客户端
            if timeout is not None and timeout != self.timeout:
                temp_client = OpenAI(
                    api_key=self.config["API_KEY"], 
                    base_url=self.config["API_BASE"],
                    timeout=timeout
                )
                client_to_use = temp_client
            else:
                client_to_use = self.client
            
            completion = client_to_use.chat.completions.create(
                model=self.config["MODEL_NAME"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # Parse response
            message_content = completion.choices[0].message.content
            
            # Extract JSON content (handle nested braces correctly)
            start = message_content.find("{")
            end = message_content.rfind("}") + 1
            
            if start == -1 or end == 0:
                return {"error": "No JSON found in response", "raw_response": message_content[:200]}
            
            json_str = message_content[start:end].strip()
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            return {"error": "Failed to parse response content", "raw_response": json_str[:200] if 'json_str' in dir() else ""}
        except Exception as e:
            return {"error": f"API error: {str(e)}"}

class OCRModalLLMResponse:
    """OCR-specific multimodal LLM response class using qwen-vl-ocr-latest model"""
    def __init__(
            self,
            api_key: str = None,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name: str = "qwen-vl-ocr-latest"
    ):
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set; create .env from .env.example")
        self.config = {
            "API_KEY": api_key,
            "API_BASE": base_url,
            "MODEL_NAME": model_name
        }
        self.client = OpenAI(api_key=self.config["API_KEY"], base_url=self.config["API_BASE"])

    def generate_response(self, image_base64: str, prompt: str, min_pixels: int = 32 * 32 * 3, max_pixels: int = 32 * 32 * 8192) -> dict:
        """
        Generate response using OCR model with image and text prompt
        
        Args:
            image_base64: Base64 encoded image string
            prompt: Text prompt for OCR task
            min_pixels: Minimum pixel threshold for image (default: 32 * 32 * 3)
            max_pixels: Maximum pixel threshold for image (default: 32 * 32 * 8192)
            
        Returns:
            Parsed JSON dictionary, returns dictionary with error key if parsing fails
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config["MODEL_NAME"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    # 输入图像的最小像素阈值，小于该值图像会放大，直到总像素大于min_pixels
                                    "min_pixels": min_pixels,
                                    # 输入图像的最大像素阈值，超过该值图像会缩小，直到总像素低于max_pixels
                                    "max_pixels": max_pixels
                                }
                            },
                            # 模型支持在以下text字段中传入Prompt，若未传入，则会使用默认的Prompt
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            # Parse response data
            message_content = completion.choices[0].message.content
            
            # Extract JSON content
            start = message_content.find("{")
            end = message_content.rfind("}") + 1
            
            if start == -1 or end == 0:
                return {"error": "No JSON found in response", "raw_response": message_content[:200]}
            
            json_str = message_content[start:end].strip()
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {str(e)}", "raw_response": message_content[:200] if 'message_content' in locals() else ""}
        except Exception as e:
            return {"error": f"API error: {str(e)}"}

class ThinkingModalLLMResponse:
    """Multimodal LLM response class with thinking mode support (for qwen3-vl-plus)"""
    def __init__(
            self,
            api_key: str = None,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name: str = "qwen3.5-plus"
    ):
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set; create .env from .env.example")
        self.config = {
            "API_KEY": api_key,
            "API_BASE": base_url,
            "MODEL_NAME": model_name
        }
        self.client = OpenAI(api_key=self.config["API_KEY"], base_url=self.config["API_BASE"])

    def generate_response(self, image_base64: str, prompt: str, enable_thinking: bool = True, verbose: bool = False) -> dict:
        """
        Generate response using thinking-enabled multimodal model
        
        Args:
            image_base64: Base64 encoded image string
            prompt: Text prompt
            enable_thinking: Whether to enable thinking mode (default: True)
            verbose: Whether to print thinking process (default: False)
            
        Returns:
            Parsed JSON dictionary with reasoning_content if thinking is enabled
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config["MODEL_NAME"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                extra_body={"enable_thinking": enable_thinking},
                stream=True,
            )
            
            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            if verbose:
                print("\n" + "-" * 20 + " Model Thinking " + "-" * 20)
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # Collect reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if verbose:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content
                
                # Collect answer content
                if hasattr(delta, "content") and delta.content:
                    if not is_answering and verbose:
                        print("\n" + "-" * 20 + " Generating JSON " + "-" * 20 + "\n")
                        is_answering = True
                    answer_content += delta.content
            
            if verbose:
                print("\n" + "-" * 50)
            
            # Parse JSON from answer_content
            start = answer_content.find("```json")
            if start != -1:
                start += 7
            else:
                start = answer_content.find("{")
            
            end = answer_content.rfind("}") + 1
            
            if start == -1 or end == 0:
                return {"error": "JSON parsing failed", "raw_response": answer_content[:500]}
            
            json_str = answer_content[start:end].strip()
            result = json.loads(json_str)
            
            # Add reasoning content to result if available
            if reasoning_content:
                result["_reasoning_content"] = reasoning_content
            
            return result
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {str(e)}", "raw_response": answer_content[:500] if 'answer_content' in locals() else ""}
        except Exception as e:
            return {"error": f"API error: {str(e)}"}

    def generate_raw_response(self, image_base64: str, prompt: str, enable_thinking: bool = True, verbose: bool = False) -> tuple:
        """
        Generate raw response without JSON parsing
        
        Returns:
            Tuple of (reasoning_content, answer_content)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config["MODEL_NAME"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                extra_body={"enable_thinking": enable_thinking},
                stream=True,
            )
            
            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            if verbose:
                print("\n" + "-" * 20 + " Model Thinking " + "-" * 20)
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if verbose:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content
                
                if hasattr(delta, "content") and delta.content:
                    if not is_answering and verbose:
                        print("\n" + "-" * 20 + " Generating Result " + "-" * 20 + "\n")
                        is_answering = True
                    answer_content += delta.content
            
            if verbose:
                print("\n" + "-" * 50)
            
            return reasoning_content, answer_content
            
        except Exception as e:
            return "", f"Error: {str(e)}"


if __name__ == "__main__":
    import sys
    llm = ModalLLMResponse(model_name="qwen-vl-max")
    # Pass an image path on the command line: python ModalLLMResponse.py /path/to/image.png
    images_path = sys.argv[1] if len(sys.argv) > 1 else "label/example.jpg"
    from Axis_extraction import encode_image
    image_base64 = encode_image(images_path)
    test_prompt = """
The graph is part of an axis and I want you to accurately identify the scale labels of x，y axis.
        Note:   The answer format is strictly JSON and requires no additional explanation. \n
                Example:{{"x/y": [0,1,2,3]}}\n
"""
    response = llm.generate_response(image_base64, test_prompt)
    # generate_response已经返回字典，无需再json.loads
    print("################")
    print("Response:", response)
    print("################")
    print('response_Y', response.get("x", 0))
    # 如果返回错误，打印详细信息
    if "error" in response:
        print("\nError details:")
        print(f"  Error code: {response.get('error_code')}")
        print(f"  Error message: {response.get('error')}")
        if "raw_response" in response:
            print(f"  Raw response: {response.get('raw_response')}")
