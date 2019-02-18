from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?authuser=0&id=1W2wmyDU77QrZjBWyzTDSCYhbHBfbPSQu&export=download'
export_file_name = 'export.pkl'

classes = ['air_jordan_1','air_jordan_2','air_jordan_3','air_jordan_4','air_jordan_5',
           'air_jordan_6','air_jordan_7','air_jordan_8','air_jordan_9','air_jordan_10',
           'air_jordan_11','air_jordan_12','air_jordan_13']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]

    # these give me a top 3 but sorted in the wrong order, I think because predictions
    # classes print in different order, so try another approach
    # _,_,losses = learn.predict(img)
    # predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    # this gives me the losses in descending order
    pred_result = learn.predict(img)[2].sort(descending=True)
    top_3_pred_probs = pred_result[0][:3]

    # convert probs to numpy array because I just want the numbers by themselves without 'tensor'
    top_3_pred_probs = top_3_pred_probs.numpy()

    # NEWEST LINE: now round the prediction probabilities from long floats to 2 decimal places
    top_3_pred_probs = [round(i, 2) for i in top_3_pred_probs]

    # grab the indices so I can use them to lookup the correct value from learn.data.classes
    top_3_pred_class_idxs = pred_result[1][:3]

    # Convert label from 'air_jordan_3' to 'Air Jordan 3' after looking up proper index
    top_3_pred_classes = [learn.data.classes[i].replace('_', ' ').title() for i in top_3_pred_class_idxs]

    predictions = list(zip(top_3_pred_classes, top_3_pred_probs))

    # return JSONResponse({'result': str(prediction)})
    return JSONResponse({'result': str(predictions)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
