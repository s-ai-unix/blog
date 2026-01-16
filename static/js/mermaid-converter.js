// 将mermaid代码块转换为可渲染的div
document.addEventListener('DOMContentLoaded', function() {
  // 查找所有代码块
  const codeBlocks = document.querySelectorAll('pre > code');

  codeBlocks.forEach((block) => {
    const code = block.textContent.trim();
    const languageClass = block.className.match(/language-(\w+)/);
    const language = languageClass ? languageClass[1] : '';

    // 检查是否是mermaid代码（通过语言标记或内容）
    if (language === 'mermaid' ||
        code.startsWith('flowchart') ||
        code.startsWith('graph') ||
        code.startsWith('gitgraph')) {

      // 创建mermaid div
      const mermaidDiv = document.createElement('div');
      mermaidDiv.className = 'mermaid';
      mermaidDiv.textContent = code;

      // 找到最外层的容器（.highlight 或 .chroma）
      let container = block.parentElement;
      while (container && !container.classList.contains('highlight') && !container.classList.contains('chroma')) {
        container = container.parentElement;
      }

      // 如果找到了容器，替换整个容器；否则只替换 pre
      if (container && (container.classList.contains('highlight') || container.classList.contains('chroma'))) {
        container.replaceWith(mermaidDiv);
      } else {
        const pre = block.parentElement;
        pre.replaceWith(mermaidDiv);
      }
    }
  });
});
