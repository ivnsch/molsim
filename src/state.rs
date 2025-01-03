use crate::{
    camera::{Camera, CameraController, CameraUniform},
    element::Element,
    instance::{AtomEntity, BondEntity, Instance, InstanceRaw},
    model::{self, DrawModel, Vertex},
    mol2_parser::{Atom, Mol, Mol2AssetLoader},
    resources, texture,
};
use cgmath::{prelude::*, Quaternion, Vector3};
use std::{collections::HashMap, iter, time::Duration};
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

    pub mol: Mol,
    pub atoms: AtomEntities,
    pub bonds: BondEntities,

    last_time: Option<Duration>, // used to calc time difference and apply physics

    // quick access
    atoms_by_id: HashMap<usize, AtomEntity>,
}

pub struct AtomEntities {
    model: model::Model,
    entities: Vec<AtomEntity>,
    buffer: wgpu::Buffer,
}

pub struct BondEntities {
    model: model::Model,
    entities: Vec<BondEntity>,
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
        // let mol = asset_loader.read("res/basic.mol2", 1).await.unwrap();
        // let mol = asset_loader.read("res/benzene.mol2", 1).await.unwrap();
        let mol = asset_loader.read("res/117_ideal.mol2", 1).await.unwrap();

        let atom_instances =
            create_atom_instances(&device, &texture_bind_group_layout, &queue, &mol).await;
        let bond_instances = create_bond_instances(
            &device,
            &texture_bind_group_layout,
            &queue,
            &mol,
            &atom_instances.entities,
        )
        .await;

        let mut atoms_by_id = HashMap::new();
        for atom in &atom_instances.entities {
            atoms_by_id.insert(atom.model.id, atom.clone());
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            mol,
            atoms: atom_instances,
            bonds: bond_instances,
            camera,
            depth_texture,
            window,
            last_time: None,
            atoms_by_id,
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
        let atom_clones = self.atoms.entities.clone();
        for atom in self.atoms.entities.iter_mut() {
            let mut total_force = Vector3::zero();
            let mass: f32 = 1.;
            for atom2_cloned in &atom_clones {
                // for now calculate lennard jones only if atoms in different molecules
                if atom.model.mol_id != atom2_cloned.model.mol_id {
                    total_force += calc_lennard_jones_force(
                        atom.instance.position,
                        atom2_cloned.instance.position,
                    );
                }
            }

            let bond_force = calc_bonds_force(atom, &self.mol, &self.atoms_by_id);
            total_force += bond_force;

            atom.instance.acceleration = total_force / mass;
            // atom.instance.acceleration /= 100.;
            atom.instance.update_physics(time_delta);
        }

        self.on_instances_updated();
    }

    /// updates instance buffer to reflect instances
    fn on_instances_updated(&mut self) {
        let atoms_instance_raw: Vec<InstanceRaw> = self
            .atoms
            .entities
            .iter()
            .map(|e| Instance::to_raw(&e.instance))
            .collect::<Vec<_>>();
        let bonds_instance_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("atoms instance Buffer"),
                    contents: bytemuck::cast_slice(&atoms_instance_raw),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        self.atoms.buffer = bonds_instance_buffer;

        self.update_instances_bonds();
        let bonds_instance_raw: Vec<InstanceRaw> = self
            .bonds
            .entities
            .iter()
            .map(|e| Instance::to_raw(&e.instance))
            .collect::<Vec<_>>();
        let bonds_instance_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bonds instance Buffer"),
                    contents: bytemuck::cast_slice(&bonds_instance_raw),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        self.bonds.buffer = bonds_instance_buffer;
    }

    fn update_instances_bonds(&mut self) {
        let new_instances = generate_instances_bonds(&self.mol, &self.atoms.entities);
        self.bonds.entities = new_instances;
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

            render_pass.set_vertex_buffer(1, self.atoms.buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.atoms.model,
                0..self.atoms.entities.len() as u32,
                &self.camera.bind_group,
            );

            render_pass.set_vertex_buffer(1, self.bonds.buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.bonds.model,
                0..self.bonds.entities.len() as u32,
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
    mol: &Mol,
) -> AtomEntities {
    let instances = generate_instances_atoms(&mol.atoms);
    create_atoms_instances_data(
        &device,
        &texture_bind_group_layout,
        &queue,
        "sphere.obj",
        instances,
    )
    .await
}

async fn create_bond_instances(
    device: &Device,
    texture_bind_group_layout: &BindGroupLayout,
    queue: &Queue,
    mol: &Mol,
    instances: &[AtomEntity],
) -> BondEntities {
    let instances = generate_instances_bonds(mol, instances);
    create_bonds_instances_data(
        &device,
        &texture_bind_group_layout,
        &queue,
        "cylinder.obj",
        instances,
    )
    .await
}

async fn create_atoms_instances_data(
    device: &Device,
    texture_bind_group_layout: &BindGroupLayout,
    queue: &Queue,
    model_file: &str,
    instances: Vec<AtomEntity>,
) -> AtomEntities {
    let model = resources::load_model(model_file, device, queue, texture_bind_group_layout)
        .await
        .unwrap();
    let instances_raw: Vec<InstanceRaw> = instances
        .iter()
        .map(|i| Instance::to_raw(&i.instance))
        .collect::<Vec<_>>();
    let instance_buffer = create_instance_buffer(device, &instances_raw);

    AtomEntities {
        model,
        entities: instances,
        buffer: instance_buffer,
    }
}
async fn create_bonds_instances_data(
    device: &Device,
    texture_bind_group_layout: &BindGroupLayout,
    queue: &Queue,
    model_file: &str,
    instances: Vec<BondEntity>,
) -> BondEntities {
    let model = resources::load_model(model_file, device, queue, texture_bind_group_layout)
        .await
        .unwrap();
    let instances_raw: Vec<InstanceRaw> = instances
        .iter()
        .map(|i| Instance::to_raw(&i.instance))
        .collect::<Vec<_>>();
    let instance_buffer = create_instance_buffer(device, &instances_raw);
    BondEntities {
        model,
        entities: instances,
        buffer: instance_buffer,
    }
}

fn create_instance_buffer(device: &Device, instances_raw: &[InstanceRaw]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(instances_raw),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn generate_instances_bonds(mol: &Mol, atoms: &[AtomEntity]) -> Vec<BondEntity> {
    // let atoms = &mol.atoms;
    mol.bonds
        .iter()
        .map(|bond| {
            // TODO make sure this doesn't crash (e.g. corrupted file)
            let atom1 = &atoms[bond.atom1 - 1].instance.position;
            let atom2 = &atoms[bond.atom2 - 1].instance.position;

            let position: Vector3<f32> = (atom1 + atom2) / 2.;

            let rotation = calc_bond_rotation(*atom1, *atom2);

            // scale cylinder mesh to distance
            // 2. is the height of the cylinder mesh
            let scale_y = atom1.distance(*atom2) / 2.;

            let instance = Instance {
                position,
                rotation,
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                scale: Vector3::new(0.1, scale_y, 0.1),
                damping: 1.,
            };

            BondEntity {
                instance,
                model: bond.clone(),
            }
        })
        .collect::<Vec<_>>()
}

fn calc_bond_rotation(point1: Vector3<f32>, point2: Vector3<f32>) -> Quaternion<f32> {
    let dir = (point2 - point1).normalize();
    let bond_up = Vector3::new(0., 1., 0.);
    if dir == bond_up {
        return Quaternion::from_angle_y(cgmath::Rad(0.0));
    } else if dir == -bond_up {
        return Quaternion::from_axis_angle(
            Vector3::new(1.0, 0.0, 0.0),
            cgmath::Rad(std::f32::consts::PI),
        );
    }

    let cross = dir.cross(bond_up).normalize();
    let dot = dir.dot(bond_up).acos();

    cgmath::Quaternion::from_axis_angle(cross, cgmath::Rad(-dot))
}

fn generate_instances_atoms(atoms: &[Atom]) -> Vec<AtomEntity> {
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

            let instance = Instance {
                position,
                rotation,
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                scale: Vector3::new(0.3, 0.3, 0.3),
                damping: 0.99,
            };

            AtomEntity {
                instance,
                model: atom.clone(),
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

    force_magnitude_to_vector(force_magnitude, pos1, pos2)
}

/// force magnitude between 2 points
fn force_magnitude_to_vector(
    magnitude: f32,
    pos1: Vector3<f32>,
    pos2: Vector3<f32>,
) -> Vector3<f32> {
    if magnitude.is_nan() || magnitude.is_infinite() {
        return Vector3::zero();
    }

    let direction = (pos2 - pos1).normalize();

    direction * magnitude
}

#[cfg(test)]
mod test {
    use cgmath::{MetricSpace, Rotation3, Vector3, Zero};

    use crate::{
        element::Element,
        instance::{AtomEntity, Instance},
        mol2_parser::Atom,
        state::{calc_bond_force, calc_lennard_jones_potential},
    };

    use super::{calc_bond_force_magnitude, calc_lennard_jones_force};

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

    #[test]
    fn test_bond_force_0() {
        // this x is the standard distance between 2 C atoms, the other atom is at position 0,
        // so distance between these atoms is the standard distance
        let atom1 = create_carbon_atom_with_x(1.526);
        let atom2 = create_carbon_atom_with_x(0.);

        let bond_length = atom1.instance.position.distance(atom2.instance.position); // sanity check to confirm expected distance
        assert_eq!(1.526, bond_length);

        // since our bond length is the standard one, it doesn't have to adjust
        // so we expect no force
        let magnitude = calc_bond_force_magnitude(&atom1, &atom2);
        assert_eq!(0., magnitude);
        let force = calc_bond_force(&atom1, &atom2);
        assert_eq!(Vector3::zero(), force);
    }

    #[test]
    fn test_bond_force_positive() {
        // these 2 atoms are at a distance slightly shorter than standard distance between 2 C atoms
        let atom1 = create_carbon_atom_with_x(-0.5);
        let atom2 = create_carbon_atom_with_x(1.);

        let bond_length = atom1.instance.position.distance(atom2.instance.position);
        assert_eq!(1.5, bond_length);

        // since the distance is shorter, we expect an expanding force (positive) to match standard distance
        let magnitude = calc_bond_force_magnitude(&atom1, &atom2);
        assert!(magnitude > 0.);

        assert_eq!(0.39743635, magnitude);

        let force = calc_bond_force(&atom1, &atom2);
        // the distance is only on x, so we expect the force to be only there too
        assert_eq!(0.39743635, force.x);
        assert_eq!(0., force.y);
        assert_eq!(0., force.z);
    }

    #[test]
    fn test_bond_force_negative() {
        // these 2 atoms are at a distance slightly larger than standard distance between 2 C atoms
        let atom1 = create_carbon_atom_with_x(-0.6);
        let atom2 = create_carbon_atom_with_x(1.);

        let bond_length = atom1.instance.position.distance(atom2.instance.position);
        assert_eq!(1.6, bond_length);

        // since the distance is larger, we expect a contracting force (negative) to match standard distance
        let magnitude = calc_bond_force_magnitude(&atom1, &atom2);
        assert!(magnitude < 0.);

        assert_eq!(-1.1311641, magnitude);

        let force = calc_bond_force(&atom1, &atom2);
        // the distance is only on x, so we expect the force to be only there too
        assert_eq!(-1.1311641, force.x);
        assert_eq!(0., force.y);
        assert_eq!(0., force.z);
    }

    /// create a cabon atom with some defaults and a specific x
    fn create_carbon_atom_with_x(x: f32) -> AtomEntity {
        AtomEntity {
            model: Atom {
                id: 1,
                name: "".to_string(),
                x,
                y: 0.,
                z: 0.,
                type_: "".to_string(),
                bond_count: 0,
                mol_name: "".to_string(),
                element: Element::C,
                mol_id: 0,
            },
            instance: Instance {
                position: Vector3 { x: x, y: 0., z: 0. },
                scale: Vector3 {
                    x: 1.,
                    y: 1.,
                    z: 1.,
                },
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                rotation: cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_z(),
                    cgmath::Deg(0.0),
                ),
                damping: 1.,
            },
        }
    }
}

/// calculates total bond energy of an atom's bonded atoms
fn calc_bonds_force(
    atom: &AtomEntity,
    mol: &Mol,
    atoms_by_id: &HashMap<usize, AtomEntity>,
) -> Vector3<f32> {
    // get neighbors (and bond: TODO check whether actually needed)
    let mut bonded_atoms = vec![];
    for bond in &mol.bonds {
        let bonded_atom = if bond.atom1 == atom.model.id {
            Some(bond.atom2)
        } else if bond.atom2 == atom.model.id {
            Some(bond.atom1)
        } else {
            None
        };

        if let Some(bonded_atom) = bonded_atom {
            // note that we assume the atom exists because this hashmap is derived from all the atoms in the molecule
            // (TODO multiple molecules)
            let atom = &atoms_by_id[&bonded_atom];
            bonded_atoms.push(atom);
        }
    }

    // TODO collapse this with previous loop
    let mut sum = Vector3::zero();
    for bonded_atom in bonded_atoms {
        let force = calc_bond_force(atom, &bonded_atom);
        sum += force
    }

    sum
}

fn calc_bond_force(atom: &AtomEntity, atom2: &AtomEntity) -> Vector3<f32> {
    let magnitude = calc_bond_force_magnitude(atom, atom2);
    let vector =
        force_magnitude_to_vector(magnitude, atom.instance.position, atom2.instance.position);

    vector
}

fn calc_bond_force_magnitude(atom1: &AtomEntity, atom2: &AtomEntity) -> f32 {
    let k = calc_bond_k(&atom1.model, &atom2.model);
    let length = atom1.instance.position.distance(atom2.instance.position);
    let length_0 = calc_bond_l0(&atom1.model, &atom2.model);

    // derivative of bond energy, which is k * (length - length_0).powi(2)
    // src (bond energy): https://youtu.be/Nd2SBfrsaw4?si=VkRbcnd1DaTaAVPK&t=107
    2. * k * (length - length_0)
}

// constant based on atoms involved
// from https://ambermd.org/antechamber/gaff.html
// table 2, column 4 or 8
fn calc_bond_k(atom1: &Atom, atom2: &Atom) -> f32 {
    match (&atom1.element, &atom2.element) {
        (Element::H, Element::H) => 4.661,
        (Element::H, Element::C) | (Element::C, Element::H) => 6.217,
        (Element::H, Element::N) | (Element::N, Element::H) => 6.057,
        (Element::C, Element::C) => 7.643,
        (Element::C, Element::O) | (Element::O, Element::C) => 7.347,
        (Element::C, Element::N) | (Element::N, Element::C) => 7.504,
        (Element::C, Element::F) | (Element::F, Element::C) => 7.227,
        // setting some fixed reasonable value for rest now. TODO complete
        _ => 6.0,
    }
}

// constant based on atoms involved
// from https://ambermd.org/antechamber/gaff.html
// table 2, column 3 or 7
fn calc_bond_l0(atom1: &Atom, atom2: &Atom) -> f32 {
    match (&atom1.element, &atom2.element) {
        (Element::H, Element::H) => 0.738,
        (Element::H, Element::C) | (Element::C, Element::H) => 1.090,
        (Element::H, Element::N) | (Element::N, Element::H) => 1.010,
        (Element::C, Element::C) => 1.526,
        (Element::C, Element::O) | (Element::O, Element::C) => 1.440,
        (Element::C, Element::N) | (Element::N, Element::C) => 1.470,
        (Element::C, Element::F) | (Element::F, Element::C) => 1.370,
        // setting some fixed reasonable value for rest now. TODO complete
        _ => 1.5,
    }
}
