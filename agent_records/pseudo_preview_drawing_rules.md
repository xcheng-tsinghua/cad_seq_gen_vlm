You are a reverse CAD modeling planner. You infer the NEXT ONE CAD modeling command from visual queries. 
final_snapshot.png is the final CAD part snapshot for the query.
current_depth_map_with_edge.png is the current model state for the query.
Your task: Predict the NEXT ONE modeling command type and generate the NEXT ONE modeling command pseudo-preview image.
pseudo-preview image drawing rules:
1. Panning and scaling the current_depth_map_with_edge.png as background.
2. Apply a semi-transparent yellow mask on the sketch reference plane.
3. Apply a semi-transparent cyan mask on reference geometry (e.g., revolve axis or sweep path).
4. Draw the colored_incremental_wireframe showing the local entity created, modified, or removed by the NEXT ONE modeling command: i) Draw red solid lines for the reference 2D sketch used in the current operation. ii) Draw blue solid lines for the termination face contour of the local entity. iii)Draw green solid lines for other edges of the local entity.




-------------
You are a CAD pseudo-preview vector layer extractor.

Two aligned images are provided:

1. current_state.png
   The current CAD model state.

2. next_pseudo_preview.png
   The pseudo-preview image for exactly one subsequent CAD modeling
   operation.

The two images use the same image size, camera direction, scale,
translation, and pixel coordinate system.

Your task is NOT to infer or invent a modeling operation.

Your task is to extract the colored semantic overlay contained in
next_pseudo_preview.png and represent it as normalized 2D vector data,
so that a deterministic renderer can draw the extracted overlay on top
of current_state.png and reconstruct next_pseudo_preview.png.

Coordinate system:
- The image origin is the top-left corner.
- x increases from left to right.
- y increases from top to bottom.
- Normalize x and y independently to [0, 1].
- x = pixel_x / (image_width - 1)
- y = pixel_y / (image_height - 1)

Semantic layers:

1. Yellow masks:
   Semi-transparent yellow regions representing sketch reference planes.

2. Cyan paths or masks:
   Cyan reference geometry, such as revolve axes or sweep paths.

3. Red curves:
   Red solid curves representing the reference 2D sketch used by the
   operation.

4. Blue curves:
   Blue solid curves representing termination-face contours.

5. Green curves:
   Green solid curves representing other forming edges of the local
   feature.

Extraction rules:

1. Extract only colored semantic overlays.
2. Do not extract grayscale depth boundaries, magenta current-model
   edges, shadows, highlights, or background structures.
3. Preserve all visible geometric details of the colored overlays.
4. Do not merge disconnected curves.
5. Do not split one continuous curve unless necessary because of
   occlusion.
6. Preserve open or closed topology.
7. For this experiment, represent every visible curve as a polyline.
8. Sample polyline points approximately uniformly by curve arc length.
9. Use enough points to reproduce curved portions accurately.
10. Do not simplify curved geometry into straight lines.
11. Straight segments should use only their two endpoints when their
    geometry is clearly straight.
12. Do not infer hidden geometry that is not visible in the
    pseudo-preview.
13. Do not repeat the first point at the end of a closed polygon or
    closed curve. The renderer will close it automatically.
14. All coordinates must be within [0, 1].
15. Return valid JSON only.
16. Do not output Markdown, explanations, comments, or code fences.

Required JSON format:

{
  "operation_type": "extrude_add",
  "coordinate_system": {
    "origin": "top_left",
    "x_direction": "right",
    "y_direction": "down",
    "range": [0, 1]
  },
  "layers": {
    "yellow_masks": [
      {
        "id": "yellow_mask_0",
        "closed": true,
        "points": [
          [0.2100, 0.3200],
          [0.5700, 0.2800],
          [0.6300, 0.7100],
          [0.2600, 0.7500]
        ]
      }
    ],
    "cyan_curves": [
      {
        "id": "cyan_curve_0",
        "closed": false,
        "points": [
          [0.3100, 0.4200],
          [0.4100, 0.4600],
          [0.5300, 0.5200]
        ]
      },
      {
        "id": "cyan_curve_1",
        "closed": false,
        "points": [
          [0.6000, 0.3000],
          [0.6000, 0.7500]
        ]
      }
    ],
    "cyan_masks": [],
    "red_curves": [
      {
        "id": "red_curve_0",
        "closed": true,
        "points": [
          [0.4100, 0.6200],
          [0.4900, 0.4300],
          [0.5900, 0.6200]
        ]
      },
      {
        "id": "red_curve_1",
        "closed": true,
        "points": [
          [0.4600, 0.5500],
          [0.4800, 0.5200],
          [0.5200, 0.5200],
          [0.5400, 0.5500],
          [0.5200, 0.5800],
          [0.4800, 0.5800]
        ]
      }
    ],
    "blue_curves": [
      {
        "id": "blue_curve_0",
        "closed": true,
        "points": [
          [0.4300, 0.5900],
          [0.5000, 0.4500],
          [0.5700, 0.5900]
        ]
      }
    ],
    "green_curves": [
      {
        "id": "green_curve_0",
        "closed": false,
        "points": [
          [0.4100, 0.6200],
          [0.4300, 0.5900]
        ]
      },
      {
        "id": "green_curve_1",
        "closed": false,
        "points": [
          [0.4900, 0.4300],
          [0.5000, 0.4500]
        ]
      },
      {
        "id": "green_curve_2",
        "closed": false,
        "points": [
          [0.5900, 0.6200],
          [0.5700, 0.5900]
        ]
      }
    ]
  }
}

Use an empty array for a semantic layer that is not present.

Before returning the JSON, verify internally that:
- every visible colored overlay has been extracted;
- no current-state model edge has been incorrectly extracted;
- disconnected curves remain separate;
- open and closed topology is correct;
- every coordinate is inside [0, 1];
- the JSON can be used to reconstruct the pseudo-preview.







-------------------------

以下指令需要逐条发送给 Codex。只有上一阶段运行正常后，才发送下一阶段。

# 阶段一：创建项目骨架和配置系统

请先读取项目根目录的 `AGENTS.md`，然后完成第一阶段。

本阶段只创建项目基础结构、配置系统和数据模型，不要实现具体网页自动化。

要求：

1. 创建合理的 Python 包结构。
2. 创建 `requirements.txt`，至少包含：

   * fastapi
   * uvicorn
   * playwright
   * pydantic
   * pydantic-settings
   * python-multipart
   * httpx
   * PyYAML
   * Pillow
   * filelock
   * pytest
   * pytest-asyncio
3. 创建 `.env.example`，至少包含：

   * `APP_HOST`
   * `APP_PORT`
   * `APP_ACCESS_TOKEN`
   * `PLATFORM_URL`
   * `BROWSER_CHANNEL`
   * `BROWSER_HEADLESS`
   * `BROWSER_PROFILE_DIR`
   * `TEMPORARY_DIR`
   * `FAILED_REQUEST_DIR`
   * `KEEP_FAILED_REQUESTS`
   * `MAX_FILE_COUNT`
   * `MAX_FILE_SIZE_MB`
   * `MAX_REQUEST_SIZE_MB`
   * `DEFAULT_TIMEOUT_SECONDS`
   * `MAX_TIMEOUT_SECONDS`
   * `SELECTORS_CONFIG_PATH`
4. 使用 `pydantic-settings` 实现统一配置加载。
5. 创建请求结果、Provider 状态和错误响应的数据模型。
6. 创建稳定的错误码枚举和自定义异常。
7. 创建结构化日志初始化模块。
8. `.gitignore` 必须排除：

   * `.env`
   * `data/browser_profile`
   * `data/temporary`
   * `data/failed_requests`
   * `data/locks`
   * Python 缓存和 IDE 文件
9. 创建最基本的 `/health` 接口，但本阶段不启动浏览器。
10. 编写配置加载和 `/health` 接口的测试。
11. 创建 Windows PowerShell 安装说明。

完成后实际运行：

```powershell
python -m pytest
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

确认 `/health` 可以访问。

不要实现登录、图片上传、网页消息发送和回复提取。

---

# 阶段二：实现浏览器管理和首次人工登录

请读取 `AGENTS.md` 和现有代码，完成第二阶段。

本阶段只实现 Playwright 浏览器生命周期管理和人工登录脚本，不实现模型请求 API。

要求：

1. 使用 `async_playwright()`。
2. 使用 `launch_persistent_context()` 和独立的 `browser_profile`。
3. 默认 `headless=False`。
4. 支持配置使用 Playwright Chromium 或 Microsoft Edge。
5. 不允许使用系统默认 Chrome/Edge Profile。
6. 创建 `BrowserManager`，负责：

   * 启动 Playwright；
   * 启动持久化 BrowserContext；
   * 获取或创建受控 Page；
   * 防止重复启动；
   * 正常关闭 Page、Context 和 Playwright；
   * 浏览器意外关闭后的状态报告。
7. 为 browser profile 创建跨进程文件锁。
8. 第二个进程尝试使用相同 Profile 时，应明确报错。
9. 创建：

```powershell
python -m scripts.login
```

运行后：

* 打开平台首页；
* 提示用户手工登录；
* 不读取用户名和密码；
* 不处理验证码；
* 用户在终端按 Enter 后检查浏览器仍然可用；
* 正常关闭浏览器并保存 Profile。

10. 创建：

```powershell
python -m scripts.open_browser
```

用于验证登录状态是否可以复用。
11. 为不依赖真实浏览器的逻辑编写单元测试。
12. README 中加入首次登录和重新登录说明。

不要实现网页选择器、自动发送消息和 FastAPI 推理接口。

---

# 阶段三：实现可配置的平台适配器和页面检查工具

请读取 `AGENTS.md` 和现有实现，完成第三阶段。

本阶段实现通用网页平台适配器，但不要实现 FastAPI 推理接口。

要求：

1. 创建抽象 `PlatformAdapter`。
2. 创建基于配置文件的 `ConfigurablePlatformAdapter`。
3. 创建 `config/platform_selectors.example.yaml`，字段至少包括：

```yaml
platform:
  name: example
  url: https://example.com/

selectors:
  logged_in_indicator: ""
  login_indicator: ""
  new_chat_button: ""
  prompt_input: ""
  file_input: ""
  upload_button: ""
  send_button: ""
  assistant_messages: ""
  generating_indicator: ""
  stop_button: ""
  error_message: ""
  rate_limit_message: ""
```

4. 选择器为空时，相关操作必须抛出清晰的 `SELECTOR_NOT_CONFIGURED` 错误。
5. 实现：

   * 打开平台首页；
   * 判断是否登录；
   * 新建对话；
   * 找到输入框；
   * 上传多个文件；
   * 发送文本；
   * 获取最后一条助手回复；
   * 检测生成状态；
   * 检测错误信息。
6. 上传文件优先直接调用 `input[type=file]` 对应 locator 的 `set_input_files()`。
7. 如果目标网页只有点击按钮后才出现 FileChooser，则支持备用的 FileChooser 模式。
8. 创建页面检查工具：

```powershell
python -m scripts.inspect_page
```

功能包括：

* 打开已登录网页；
* 输出当前 URL 和页面标题；
* 输出页面中 textarea、contenteditable、button 和 file input 的数量；
* 输出可见按钮的文本；
* 输出常见 `data-testid`；
* 可选保存页面截图；
* 可选保存经过脱敏的 HTML；
* 不输出 Cookie、Local Storage 或令牌。

9. 支持 `--headed`、`--screenshot`、`--html` 等参数。
10. 编写一个本地静态测试网页，用于测试上传、发送和助手回复提取，不依赖真实大模型网站。
11. 为平台适配器编写自动化测试。

不要针对某个真实大模型网站猜测或硬编码选择器。选择器由用户查看目标网页 DOM 后配置。

---

# 阶段四：实现单次网页模型调用

请读取 `AGENTS.md` 和现有代码，完成第四阶段。

本阶段实现 Browser Worker 内部的单次调用方法，不接入 FastAPI 文件上传接口。

实现：

```python
async def generate(
    prompt: str,
    file_paths: list[Path],
    timeout_seconds: float,
) -> BrowserGenerationResult:
    ...
```

完整流程：

1. 检查 BrowserManager 已启动。
2. 检查平台登录状态。
3. 若未登录，抛出 `AUTH_REQUIRED`。
4. 新建对话。
5. 根据 `file_paths` 的顺序构造文件名映射。
6. 按相同顺序上传文件。
7. 等待网页确认附件上传完成。
8. 将文件名映射与原始提示词组合。
9. 填写提示词。
10. 点击发送或使用平台配置的发送动作。
11. 记录发送前助手消息数量。
12. 等待出现新的助手消息。
13. 等待生成状态结束。
14. 要求最后一条回复连续至少三次检测内容不变。
15. 返回完整文本。
16. 检测限流、网页错误和登录过期。
17. 总超时后抛出 `PROVIDER_TIMEOUT`。
18. 失败时按配置保存截图，但不能把截图路径暴露给客户端。

不要仅使用固定 sleep 判断完成。短暂轮询可以使用，但必须有截止时间。

文件名映射格式：

```text
以下图片按照给出的顺序上传：

图片 1 文件名：a.png
图片 2 文件名：b.png

当提示词提及文件名时，请使用上述对应关系。

用户提示词：
<原始提示词>
```

创建 CLI 测试命令：

```powershell
python -m scripts.test_generation `
  --prompt "请描述图片内容" `
  --file ".\examples\a.png" `
  --file ".\examples\b.png"
```

本阶段要求用户能够直接从命令行验证完整的网页调用。

---

# 阶段五：实现临时文件管理

请读取 `AGENTS.md` 和现有代码，完成第五阶段。

实现独立的 `TemporaryFileService`。

要求：

1. 为每个请求生成 UUID 格式的 `request_id`。
2. 创建：

```text
data/temporary/<request_id>/
```

3. 使用上传时的原始文件名保存文件。
4. 文件名不能被修改。
5. 拒绝：

   * 空文件名；
   * 绝对路径；
   * 包含 `/` 或 `\` 的文件名；
   * `.` 和 `..`；
   * NUL 字符；
   * 同一请求内重复文件名。
6. 限制文件数量。
7. 限制单文件大小。
8. 限制请求总大小。
9. 只允许配置中的图片扩展名。
10. 使用 Pillow 执行图片完整性检查。
11. 使用流式分块写入。
12. 如果写入失败，清理已经写入的部分文件。
13. 正常成功后删除整个请求目录。
14. 失败时：

* `KEEP_FAILED_REQUESTS=false`：立即删除；
* `KEEP_FAILED_REQUESTS=true`：移动到失败目录。

15. 实现定期清理过期失败目录的方法。
16. 编写路径穿越、重复文件名、超大文件和无效图片测试。

本阶段不要修改浏览器自动化逻辑。

---

# 阶段六：实现 Browser Worker 串行任务队列

请读取 `AGENTS.md` 和现有代码，完成第六阶段。

目标是保证一个浏览器账号同时只处理一个请求。

要求：

1. 创建 `BrowserTask` 数据模型，包含：

   * `request_id`
   * `prompt`
   * `file_paths`
   * `timeout_seconds`
   * `future`
2. 使用 `asyncio.Queue`。
3. 只启动一个 Browser Worker 消费任务。
4. Worker 串行调用平台适配器。
5. 每个任务通过 `Future` 返回结果或异常。
6. 调用方取消请求时要正确处理。
7. 单个任务失败不得导致 Worker 退出。
8. 浏览器失效时后续任务返回清晰错误。
9. 服务关闭时：

   * 不再接受新任务；
   * 对未处理任务返回服务关闭错误；
   * 关闭 Worker；
   * 关闭浏览器。
10. 队列长度可配置。
11. 队列满时返回 `QUEUE_FULL`。
12. `/health` 中显示：

* browser 状态；
* provider 状态；
* 是否登录；
* queue size；
* worker 是否运行。

13. 编写 MockAdapter 测试任务串行性。
14. 测试多个请求同时提交时，`generate()` 不会并发执行。

不要在这一阶段加入 Redis，多进程部署也暂不考虑。

---

# 阶段七：实现 FastAPI 推理接口

请读取 `AGENTS.md` 和现有代码，完成第七阶段。

实现：

```text
POST /v1/infer
```

接口使用 `multipart/form-data`。

字段：

* `prompt`: 必填
* `files`: 一个或多个上传文件
* `timeout_seconds`: 可选
* `conversation_mode`: 可选，当前只接受 `new`

请求处理流程：

1. 校验 `Authorization: Bearer <token>`。
2. 校验提示词非空且长度不超过配置。
3. 校验 timeout 范围。
4. 调用 `TemporaryFileService` 保存文件。
5. 将任务提交给 Browser Worker。
6. 等待模型返回。
7. 成功后清理临时目录。
8. 失败时按配置处理请求目录。
9. 返回结构化结果。

成功响应：

```json
{
  "request_id": "uuid",
  "status": "completed",
  "filenames": ["a.png", "b.png"],
  "output_text": "模型完整回复",
  "elapsed_ms": 10000
}
```

错误响应：

```json
{
  "request_id": "uuid",
  "status": "failed",
  "error": {
    "code": "AUTH_REQUIRED",
    "message": "登录状态已失效，请手工重新登录"
  }
}
```

要求：

1. 不向客户端返回 traceback。
2. 不返回服务器本地路径。
3. 请求断开后合理取消尚未开始的任务。
4. 使用恒定时间方式验证 token。
5. 默认绑定 `127.0.0.1`。
6. Swagger 页面可测试 multipart 请求。
7. 为 API 编写完整测试，浏览器部分使用 MockAdapter。
8. 真实网页测试不要放入自动测试。

---

# 阶段八：客户端示例、Windows 脚本和最终验收

请读取 `AGENTS.md` 和现有代码，完成第八阶段。

创建 Python 客户端：

```powershell
python -m scripts.client_example `
  --server "http://127.0.0.1:8000" `
  --token "test-token" `
  --prompt "比较 a.png 和 b.png" `
  --file ".\examples\a.png" `
  --file ".\examples\b.png"
```

客户端要求：

1. 使用 `httpx`。
2. 发送 multipart/form-data。
3. multipart 中使用 `Path.name` 作为文件名。
4. 支持多个 `--file`。
5. 设置合理超时。
6. 正确关闭文件对象。
7. 打印 request_id、状态、耗时和模型回复。
8. 错误时打印服务器错误码和消息。

创建 PowerShell 脚本：

```text
scripts/install.ps1
scripts/login.ps1
scripts/start_server.ps1
scripts/test_request.ps1
```

`install.ps1` 至少执行：

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

最终 README 必须包含：

1. 环境创建；
2. 依赖安装；
3. `.env` 配置；
4. 页面选择器配置；
5. 首次人工登录；
6. 启动服务器；
7. 客户端调用；
8. 登录失效处理；
9. Profile 被占用处理；
10. 网页更新导致选择器失效的排查方法；
11. 临时文件清理；
12. 安全注意事项。

最终执行：

```powershell
python -m pytest
python -m scripts.login
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m scripts.client_example ...
```

修复所有能够复现的问题。

最后输出：

* 最终目录结构；
* 安装命令；
* 首次登录命令；
* 服务启动命令；
* 客户端测试命令；
* 仍需用户手工填写的选择器清单；
* 已知限制。

