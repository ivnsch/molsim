use std::{iter, time::Duration};

use crate::{
    camera::{Camera, CameraController, CameraUniform},
    instance::{Instance, InstanceEntity, InstanceRaw},
    model::{self, DrawModel, Vertex},
    mol2_parser::{Atom, Mol, Mol2AssetLoader},
    resources, texture,
};
use cgmath::{prelude::*, Vector3};
use wgpu::{
    util::DeviceExt, Adapter, BindGroup, BindGroupLayout, Buffer, Device, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPipeline, ShaderModule,
    Surface, SurfaceConfiguration, TextureView,
};
use winit::{dpi::PhysicalSize, event::*, window::Window};

pub struct State<'a> {
    pub surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera: CameraDeps,
    pub depth_texture: texture::Texture,
    pub window: &'a Window,
    pub atom_instances: Instances,

    last_time: Option<Duration>, // used to calc time difference and apply physics
}

pub struct Instances {
    model: model::Model,
    instances: Vec<Instance>,
    buffer: wgpu::Buffer,
}

impl<'a> State<'a> {
    pub async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        log::warn!("WGPU setup");

        let instance = create_wgpu_instance();

        let surface = instance.create_surface(window).unwrap();
        let adapter = create_adapter(instance, &surface).await;

        log::warn!("device and queue");

        let (device, queue) = get_device_and_queue(&adapter).await;
        let config = create_config(&adapter, &surface, size);

        let texture_bind_group_layout = create_texture_bind_group_layout(&device);

        let camera = create_camera_deps(&device, &config);

        log::warn!("Load model");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera.bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            create_render_pipeline(&device, render_pipeline_layout, shader, &config);

        let asset_loader = Mol2AssetLoader {};
        let mol = asset_loader.read("res/benzene.mol2", 1).await.unwrap();

        let atom_instances =
            create_atom_instances(&device, &texture_bind_group_layout, &queue, mol).await;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            atom_instances,
            camera,
            depth_texture,
            window,
            last_time: None,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.size = new_size;
            self.camera.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.controller.process_events(event)
    }

    pub fn update(&mut self, time: Duration) {
        self.camera
            .controller
            .update_camera(&mut self.camera.camera);
        log::info!("{:?}", self.camera.camera);
        self.camera.uniform.update_view_proj(&self.camera.camera);
        log::info!("{:?}", self.camera.uniform);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );

        self.move_instances(time);
    }

    fn move_instances(&mut self, time: Duration) {
        match self.last_time {
            Some(last_time) => {
                let time_delta = time - last_time;
                self.move_instances_with_time_delta(time_delta)
            }
            None => {}
        }
        self.last_time = Some(time);
    }

    fn move_instances_with_time_delta(&mut self, time_delta: Duration) {
        // just some arbitrary motion

        // TODO more performant way to do nested loop with mutability
        let clone = self.atom_instances.instances.clone();
        for instance in self.atom_instances.instances.iter_mut() {
            let mut total_force = Vector3::zero();
            let mass: f32 = 1.;
            for instance2 in &clone {
                let calculate_lennard_potential = match (&instance.entity, &instance2.entity) {
                    (InstanceEntity::Atom(atom1), InstanceEntity::Atom(atom2)) => {
                        atom1.mol_id != atom2.mol_id
                    }
                };
                if calculate_lennard_potential {
                    total_force += calc_lennard_jones_force(instance.position, instance2.position);
                }
            }
            instance.acceleration = total_force / mass;
            instance.update_physics(time_delta);
        }

        self.on_instances_updated();
    }

    /// updates instance buffer to reflect instances
    fn on_instances_updated(&mut self) {
        let instance_data: Vec<InstanceRaw> = self
            .atom_instances
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.atom_instances.buffer = instance_buffer;
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(render_color_attachement(&view))],
                depth_stencil_attachment: Some(render_depth_stencil_attachement(
                    &self.depth_texture.view,
                )),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.atom_instances.buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.atom_instances.model,
                0..self.atom_instances.instances.len() as u32,
                &self.camera.bind_group,
            );
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn render_color_attachement<'a>(view: &'a TextureView) -> RenderPassColorAttachment<'a> {
    RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn render_depth_stencil_attachement<'a>(
    view: &'a TextureView,
) -> RenderPassDepthStencilAttachment<'a> {
    RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}

async fn get_device_and_queue(adapter: &Adapter) -> (Device, Queue) {
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
            },
            // Some(&std::path::Path::new("trace")), // Trace path
            None, // Trace path
        )
        .await
        .unwrap();
    (device, queue)
}

async fn create_adapter<'a>(instance: wgpu::Instance, surface: &'a Surface<'_>) -> Adapter {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap()
}

// The instance is a handle to our GPU
// BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
fn create_wgpu_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        #[cfg(not(target_arch = "wasm32"))]
        backends: wgpu::Backends::PRIMARY,
        #[cfg(target_arch = "wasm32")]
        backends: wgpu::Backends::GL,
        ..Default::default()
    })
}

fn create_config<'a>(
    adapter: &Adapter,
    surface: &'a Surface<'_>,
    size: PhysicalSize<u32>,
) -> SurfaceConfiguration {
    let surface_caps = surface.get_capabilities(&adapter);
    // Shader code in this tutorial assumes an Srgb surface texture. Using a different
    // one will result all the colors comming out darker. If you want to support non
    // Srgb surfaces, you'll need to account for that when drawing to the frame.
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);
    SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

fn create_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("texture_bind_group_layout"),
    })
}

pub struct CameraDeps {
    pub camera: Camera,
    pub controller: CameraController,
    pub uniform: CameraUniform,
    pub buffer: Buffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

fn create_camera_deps(device: &Device, config: &SurfaceConfiguration) -> CameraDeps {
    let camera = Camera {
        eye: (0.0, 15.0, -30.0).into(),
        target: (0.0, 0.0, 0.0).into(),
        up: cgmath::Vector3::unit_y(),
        aspect: config.width as f32 / config.height as f32,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };
    let controller = CameraController::new(0.2);

    let mut uniform = CameraUniform::new();
    uniform.update_view_proj(&camera);

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("camera_bind_group_layout"),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
        label: Some("camera_bind_group"),
    });

    CameraDeps {
        camera,
        controller,
        uniform,
        buffer,
        bind_group_layout,
        bind_group,
    }
}

async fn create_atom_instances(
    device: &Device,
    texture_bind_group_layout: &BindGroupLayout,
    queue: &Queue,
    mol: Mol,
) -> Instances {
    let instances = generate_instances(mol.atoms);
    create_instances_data(
        &device,
        &texture_bind_group_layout,
        &queue,
        "sphere.obj",
        instances,
    )
    .await
}

async fn create_instances_data(
    device: &Device,
    texture_bind_group_layout: &BindGroupLayout,
    queue: &Queue,
    model_file: &str,
    instances: Vec<Instance>,
) -> Instances {
    let model = resources::load_model(model_file, device, queue, texture_bind_group_layout)
        .await
        .unwrap();
    let instance_raw: Vec<InstanceRaw> = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instance_raw),
        usage: wgpu::BufferUsages::VERTEX,
    });

    Instances {
        model,
        instances,
        buffer: instance_buffer,
    }
}

fn generate_instances(atoms: Vec<Atom>) -> Vec<Instance> {
    atoms
        .into_iter()
        .map(|atom| {
            let position = cgmath::Vector3 {
                x: atom.x,
                y: atom.y,
                z: atom.z,
            };
            // println!("added atom at: {:?}", position);

            let rotation = if position.is_zero() {
                cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
            } else {
                cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
            };

            Instance {
                position,
                rotation,
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                scale: 0.3,
                entity: InstanceEntity::Atom(atom),
            }
        })
        .collect::<Vec<_>>()
}

fn create_render_pipeline(
    device: &Device,
    render_pipeline_layout: wgpu::PipelineLayout,
    shader: ShaderModule,
    config: &SurfaceConfiguration,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
            // or Features::POLYGON_MODE_POINT
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: texture::Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
        // Useful for optimizing shader compilation on Android
        cache: None,
    })
}

// used in test
#[allow(unused)]
fn calc_lennard_jones_potential(sigma: f32, epsilon: f32, distance: f32) -> f32 {
    let fraction = sigma / distance;
    4. * epsilon * (fraction.powi(12) - fraction.powi(6))
}

fn calc_lennard_jones_potential_derivative(sigma: f32, epsilon: f32, distance: f32) -> f32 {
    let fraction = sigma / distance;
    24. * epsilon / distance * (fraction.powi(6) - 2. * fraction.powi(12))
}

fn calc_lennard_jones_force(pos1: Vector3<f32>, pos2: Vector3<f32>) -> Vector3<f32> {
    let sigma = 2.5;
    let epsilon = 0.001; // eV

    let distance = pos1.distance(pos2);
    if distance.round() == 0.0 {
        return Vector3::zero(); // avoid division by zero
    }

    // -1 is correct, but particles just keep drifting apart with it, while 1 causes more expected attraction/repulsion behavior
    // something is wrong elsewhere?
    // let force_magnitude = -1. * calc_lennard_jones_potential_derivative(sigma, epsilon, distance);
    let force_magnitude = 1. * calc_lennard_jones_potential_derivative(sigma, epsilon, distance);

    if force_magnitude.is_nan() || force_magnitude.is_infinite() {
        return Vector3::zero();
    }

    let direction = (pos2 - pos1).normalize();

    direction * force_magnitude
}

#[cfg(test)]
mod test {
    use cgmath::{MetricSpace, Vector3};

    use crate::state::calc_lennard_jones_potential;

    use super::calc_lennard_jones_force;

    // to get an idea of magnitudes
    #[test]
    fn print_lennard_jones_potential_at_different_distances() {
        let pos1 = Vector3::new(0., 0., 0.);

        let positions2 = vec![
            Vector3::new(7., 0., 0.),  // -7.085309e-6
            Vector3::new(4., 0., 0.),  // -0.0003
            Vector3::new(2., 0., 0.),  // 0.30
            Vector3::new(1., 0., 0.),  // 2855
            Vector3::new(0.5, 0., 0.), // 23436750
        ];

        let sigma = 2.5;
        let epsilon = 0.001; // eV

        for pos2 in positions2 {
            let distance = pos1.distance(pos2);
            let potential = calc_lennard_jones_potential(sigma, epsilon, distance);
            let force = calc_lennard_jones_force(pos1, pos2);
            println!(
                "distance: {:?}, potential: {:?}, force: {:?}",
                distance, potential, force
            );
        }
    }
}
