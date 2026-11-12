# salad-h3 trial image: Salad's comfyui-api stack (CUDA 12.8 / torch 2.8,
# known-good on Blackwell) as the base, pinned BY DIGEST, with the stock
# wrapper unused. We add: pinned ComfyUI v0.45.0 (tarball checksum-verified;
# the version the verified Modal reference run used) and our own
# entrypoint (status server + verified downloader + native-API proxy).
# The image contains NO weights, NO input images, NO prompts and NO
# secrets, so it is published as a public image, itself pinned by digest.
FROM ghcr.io/saladtechnologies/comfyui-api@sha256:8345d9d1d77c6c779eb55d69abaee6947f869f3dc044509868abf19cfce71860

ADD --checksum=sha256:408670ede5da641b30a2b3f1e39e5b84d75c12357e75695416a6e6cc5bcbc4bb https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/v0.45.0.tar.gz /tmp/comfyui.tar.gz
RUN mkdir -p /opt/comfyui-h3 \
 && tar -xzf /tmp/comfyui.tar.gz -C /opt/comfyui-h3 --strip-components=1 \
 && rm /tmp/comfyui.tar.gz \
 && python3 -m pip install --no-cache-dir -r /opt/comfyui-h3/requirements.txt \
 && python3 -c "import torch; assert torch.__version__.startswith('2.8'), torch.__version__; print('torch pinned OK:', torch.__version__)" \
 && test -f /opt/comfyui-h3/nodes.py && echo "comfyui v0.45.0 in place"

COPY entrypoint.py weights.json /opt/h3/
RUN python3 -m py_compile /opt/h3/entrypoint.py && echo "entrypoint compiles" \
 && mkdir -p /opt/comfyui-h3/models /opt/comfyui-h3/input /opt/comfyui-h3/output \
 && chmod -R a+rwX /opt/comfyui-h3 /opt/h3

ENTRYPOINT ["python3", "/opt/h3/entrypoint.py"]
