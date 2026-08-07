# 通用工具,例如图片处理,文件操作等
import base64#用于图片编码和解码
def base64_encode_image(image_path)->str:
    """
    将本地图片转化为base64字符串,用于多模态图文的一个请求
    :param image_path: 图片路径
    :return: base64编码后的字符串
    """
    #在这里rb表示二进制读取,确保图片数据的完整性和准确性
    with open(image_path, "rb") as f:
    #读取图片数据
        image_data = f.read()
    #将二进制数据编码为base64字符串
    return base64.b64encode(image_data).decode("utf-8")

