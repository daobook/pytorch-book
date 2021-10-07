# PyTorch 常见 Bug

## urllib.error.HTTPError: HTTP Error 403: rate limit exceeded when loading resnet18 from pytorch hub

这是 Pytorch 1.9 中的一个错误。作为解决方法，请尝试添加：

```python
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
```