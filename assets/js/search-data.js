// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "What I cannot create, I don&#39;t understand",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "This is a description of the page. You can modify it in &#39;_pages/cv.md&#39;. You can also change or remove the top pdf download button.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-compression-via-denoising",
        
          title: "Compression via denoising",
        
        description: "Idealistic models that inspire deep network structures",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/diffusion-denoising/";
          
        },
      },{id: "post-optimization-from-basics-to-ista",
        
          title: "Optimization from basics to ISTA",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/optimization-basics/";
          
        },
      },{id: "post-what-to-learn-and-introduction-to-problems-solvable-by-analytical-approach",
        
          title: "What to learn and introduction to problems solvable by analytical approach",
        
        description: "Idealistic models that inspire deep network structures",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/what-to-learn/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-project-whitebox-transformer-implementation-is-published",
          title: 'Project: Whitebox Transformer Implementation is published',
          description: "",
          section: "News",},{id: "news-project-resemblance-of-cross-attention-like-operator-with-conditional-gmm-denoiser",
          title: 'Project: Resemblance of Cross Attention like Operator with Conditional GMM Denoiser',
          description: "",
          section: "News",},{id: "projects-whitebox-transformer-implementation",
          title: 'Whitebox Transformer Implementation',
          description: "Interpretable deep learning architecture and data representation",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-resemblance-of-cross-attention-like-operator-with-condional-gmm-denoiser",
          title: 'Resemblance of Cross Attention like Operator with Condional GMM Denoiser',
          description: "Demystify cross attention mechanism",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6B%61%6C%6F%6B%73%65%72%69%6F%75%73@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
