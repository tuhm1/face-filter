import * as FaceDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
    Box3,
    Matrix4,
    OrthographicCamera,
    PointLight,
    Scene,
    Vector3,
    WebGLRenderer
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

renderPage();

async function renderPage() {
    const pGetVideoStream = getVideoStream();
    const filterNames = [
        'Hearts',
        'Thug Life',
        'Bloody Eyes',
        'Guy Fawkes'
    ];
    const pLoadFilters = [
        loadHeartFilter,
        loadThugLifeFilter,
        loadBloodyEyesFilter,
        loadGuyFawkesFilter
    ];
    const pInitFilters = pLoadFilters.map(load => load());

    const page = document.createElement('div');
    page.classList.add('page');

    const preview = document.createElement('div');
    preview.classList.add('preview');
    page.appendChild(preview);

    const buttons = document.createElement('div');
    buttons.classList.add('buttons');

    const filterButtonsDiv = document.createElement('div');
    filterButtonsDiv.classList.add('filters');

    buttons.appendChild(filterButtonsDiv);

    const record = document.createElement('div');

    const recordButton = document.createElement('button');
    recordButton.textContent = 'Record';
    record.appendChild(recordButton);
    buttons.appendChild(record);
    page.appendChild(buttons);
    document.body.appendChild(page);

    const filterButtons = filterNames.map(name => {
        const button = document.createElement('button');
        button.textContent = name;
        filterButtonsDiv.appendChild(button);
        return button;
    });

    pGetVideoStream.then(({ canvas, renderer, settings, setFilter }) => {
        canvas.classList.add('canvas');
        preview.appendChild(canvas);

        recordButton.onclick = async () => {
            const stop = recordCanvas(canvas);
            recordButton.remove();
            const stopButton = document.createElement('button');
            stopButton.textContent = 'Stop';
            stopButton.addEventListener('click', async () => {
                const blob = await stop();
                downloadBlob(blob, 'face.webm');
                stopButton.remove();
                record.appendChild(recordButton);
            });
            record.appendChild(stopButton);
        }

        filterButtons.forEach(async (button, i) => {
            const initFilter = await pInitFilters[i];
            const filter = initFilter(renderer, settings);
            button.addEventListener('click', () => {
                setFilter(filter);
            });
        });
    });
}

function getVideoStream() {
    return navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            const video = document.createElement('video');
            video.autoplay = true;
            video.srcObject = stream;
            const pVideoStarted = new Promise(resolve => video.onplay = resolve);

            const settings = stream.getVideoTracks()[0].getSettings();

            const canvas = document.createElement('canvas');
            canvas.width = settings.width;
            canvas.height = settings.height;

            const renderer = new WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(settings.width, settings.height);

            const context = canvas.getContext('2d');
            let currentFilter;
            const setFilter = newFilter => {
                currentFilter = newFilter;
            };
            const update = async () => {
                currentFilter && await currentFilter(video);
                context.drawImage(video, 0, 0);
                context.drawImage(renderer.domElement, 0, 0);
                requestAnimationFrame(update);
            }
            pVideoStarted.then(update);
            return { canvas, renderer, settings, setFilter };
        });
}

let pFaceDetectionModel;
async function loadFaceDetectionModel() {
    if (!pFaceDetectionModel) {
        const webgl2 = document.createElement('canvas').getContext('webgl2');
        if (webgl2) {
            await tf.setBackend('webgl');
        } else {
            setWasmPaths('/tfjs-backend-wasm/');
            await tf.setBackend('wasm');
        }
        pFaceDetectionModel = FaceDetection.load(FaceDetection.SupportedPackages.mediapipeFacemesh);
    }
    return pFaceDetectionModel;
}

async function loadThugLifeFilter() {
    const pModel = loadFaceDetectionModel();
    const pGltf = new GLTFLoader().loadAsync('/3d-objects/thug_life_glasses/scene.gltf');
    const [model, gltf] = await Promise.all([pModel, pGltf]);
    return function initFilter(renderer, settings) {
        const scene = new Scene();

        const width = settings.width;
        const height = settings.height;

        const camera = new OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -1000, 1000);
        camera.position.set(width / 2, -height / 2, height / 2);
        const light = new PointLight();
        light.position.copy(camera.position);
        scene.add(light);

        const maskOrg = gltf.scene;
        const box = new Box3().setFromObject(maskOrg);
        const maskSize = box.getSize(new Vector3());
        const offset = box.max.z;

        return async function render(video) {
            const faces = await model.estimateFaces({ input: video });
            const masks = [];
            faces.forEach(async (face) => {
                const mask = gltf.scene.clone();
                const meshPointToVec3 = ([x, y, z]) => new Vector3(x, -y, -z);
                const left = meshPointToVec3(face.scaledMesh[454]);
                const right = meshPointToVec3(face.scaledMesh[234]);
                const top = meshPointToVec3(face.annotations.midwayBetweenEyes[0]);
                const bottom = meshPointToVec3(face.annotations.noseBottom[0]);
                const rightLeft = left.clone().sub(right);
                const bottomTop = top.clone().sub(bottom);
                const cross = rightLeft.clone().cross(bottomTop);
                const matrix = new Matrix4().makeBasis(
                    rightLeft.clone().normalize(),
                    bottomTop.clone().normalize(),
                    cross.clone().normalize()
                );
                mask.setRotationFromMatrix(matrix);
                mask.position.copy(top);
                const scale = rightLeft.length() / maskSize.x;
                mask.scale.setScalar(scale);
                mask.translateOnAxis(cross.clone().normalize(), -offset * scale);

                scene.add(mask);
                masks.push(mask);
            });

            renderer.render(scene, camera);

            masks.forEach(mask => {
                scene.remove(mask);
            });
        }
    }
}

async function loadHeartFilter() {
    const pModel = loadFaceDetectionModel();
    const pGltf = new GLTFLoader().loadAsync('/3d-objects/heart_glasses/scene.gltf');
    const [model, gltf] = await Promise.all([pModel, pGltf]);
    return function initFilter(renderer, settings) {
        const scene = new Scene();

        const width = settings.width;
        const height = settings.height;

        const camera = new OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -1000, 1000);
        camera.position.set(width / 2, -height / 2, height / 2);

        const light = new PointLight();
        light.position.copy(camera.position);
        scene.add(light);

        const maskOrg = gltf.scene;
        const box = new Box3().setFromObject(maskOrg);
        const maskSize = box.getSize(new Vector3());
        const offset = box.max.z;

        return async function render(video) {
            const faces = await model.estimateFaces({ input: video });
            const masks = [];
            faces.forEach(async (face) => {
                const mask = gltf.scene.clone();
                const meshPointToVec3 = ([x, y, z]) => new Vector3(x, -y, -z);
                const left = meshPointToVec3(face.scaledMesh[454]);
                const right = meshPointToVec3(face.scaledMesh[234]);
                const top = meshPointToVec3(face.annotations.midwayBetweenEyes[0]);
                const bottom = meshPointToVec3(face.annotations.noseBottom[0]);
                const rightLeft = left.clone().sub(right);
                const bottomTop = top.clone().sub(bottom);
                const cross = rightLeft.clone().cross(bottomTop);
                const rotationMatrix = new Matrix4().makeBasis(
                    rightLeft.clone().normalize(),
                    bottomTop.clone().normalize(),
                    cross.clone().normalize()
                );
                mask.setRotationFromMatrix(rotationMatrix);
                mask.position.copy(top);
                const scale = rightLeft.length() / maskSize.x;
                mask.scale.setScalar(scale);
                mask.translateOnAxis(cross.clone().normalize(), -offset * scale);

                scene.add(mask);
                masks.push(mask);
            });

            renderer.render(scene, camera);

            masks.forEach(mask => {
                scene.remove(mask);
            });
        };
    }
}

async function loadGuyFawkesFilter() {
    const pModel = loadFaceDetectionModel();
    const pGltf = new GLTFLoader().loadAsync('/3d-objects/guy_fawkes/scene.gltf');
    const [model, gltf] = await Promise.all([pModel, pGltf]);
    return function initFilter(renderer, settings) {
        const scene = new Scene();

        const width = settings.width;
        const height = settings.height;

        const camera = new OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -1000, 1000);
        camera.position.set(width / 2, -height / 2, height / 2);

        const light = new PointLight();
        light.position.copy(camera.position);
        scene.add(light);

        const maskOrg = gltf.scene;
        const box = new Box3().setFromObject(maskOrg);
        const maskSize = box.getSize(new Vector3());

        return async function render(video) {
            const faces = await model.estimateFaces({ input: video });
            const masks = [];
            faces.forEach(async (face) => {
                const mask = gltf.scene.clone();
                const meshPointToVec3 = ([x, y, z]) => new Vector3(x, -y, -z);
                const left = meshPointToVec3(face.scaledMesh[454]);
                const right = meshPointToVec3(face.scaledMesh[234]);
                const top = meshPointToVec3(face.scaledMesh[10]);
                const bottom = meshPointToVec3(face.scaledMesh[152]);
                const rightLeft = left.clone().sub(right);
                const topBottom = bottom.clone().sub(top);
                const cross = topBottom.clone().cross(rightLeft);
                const rotationMatrix = new Matrix4().makeBasis(
                    rightLeft.clone().normalize(),
                    cross.clone().normalize(),
                    topBottom.clone().normalize()
                );
                mask.setRotationFromMatrix(rotationMatrix);
                const position = top.clone().addScaledVector(topBottom, 0.5);
                mask.position.copy(position);
                const scale = rightLeft.length() / maskSize.x;
                mask.scale.setScalar(scale);
                scene.add(mask);
                masks.push(mask);
            });

            renderer.render(scene, camera);

            masks.forEach(mask => {
                scene.remove(mask);
            });
        };
    }
}

async function loadBloodyEyesFilter() {
    const pModel = loadFaceDetectionModel();
    const pGltf = new GLTFLoader().loadAsync('/3d-objects/bloody_eyes_mask/scene.gltf');
    const [model, gltf] = await Promise.all([pModel, pGltf]);
    return function initFilter(renderer, settings) {
        const scene = new Scene();

        const width = settings.width;
        const height = settings.height;

        const camera = new OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -1000, 1000);
        camera.position.set(width / 2, -height / 2, height / 2);

        const light = new PointLight();
        light.position.copy(camera.position);
        scene.add(light);

        const maskOrg = gltf.scene;
        const box = new Box3().setFromObject(maskOrg);
        const maskSize = box.getSize(new Vector3());

        return async function render(video) {
            const faces = await model.estimateFaces({ input: video });
            const masks = [];
            faces.forEach(async (face) => {
                const mask = gltf.scene.clone();
                const meshPointToVec3 = ([x, y, z]) => new Vector3(x, -y, -z);
                const left = meshPointToVec3(face.scaledMesh[454]);
                const right = meshPointToVec3(face.scaledMesh[234]);
                const top = meshPointToVec3(face.scaledMesh[10]);
                const bottom = meshPointToVec3(face.scaledMesh[152]);
                const rightLeft = left.clone().sub(right);
                const bottomTop = top.clone().sub(bottom);
                const cross = rightLeft.clone().cross(bottomTop);
                const rotationMatrix = new Matrix4().makeBasis(
                    rightLeft.clone().normalize(),
                    bottomTop.clone().normalize(),
                    cross.clone().normalize()
                );
                mask.setRotationFromMatrix(rotationMatrix);
                mask.position.copy(bottom);
                const scale = rightLeft.length() / maskSize.x;
                mask.scale.setScalar(scale);
                scene.add(mask);
                masks.push(mask);
            });

            renderer.render(scene, camera);

            masks.forEach(mask => {
                scene.remove(mask);
            });
        };
    }
}

function recordCanvas(canvas) {
    const stream = canvas.captureStream();
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
    const chunks = [];
    recorder.ondataavailable = e => {
        e.data?.size > 0 && chunks.push(e.data);
    }
    recorder.start();
    return () => new Promise(resolve => {
        recorder.onstop = e => {
            const blob = new Blob(chunks);
            resolve(blob);
        }
        recorder.stop();
    });
}

function downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.onclick = () => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    };
}